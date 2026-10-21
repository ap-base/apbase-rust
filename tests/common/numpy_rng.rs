#![allow(dead_code)]

use super::ziggurat::{FI_DOUBLE, KI_DOUBLE, WI_DOUBLE, ZIGGURAT_NOR_INV_R, ZIGGURAT_NOR_R};

const PCG_MULTIPLIER: u128 = (2_549_297_995_355_413_924u128 << 64) + 4_865_540_595_714_422_341u128;

fn seed_sequence_state(seed: u64) -> [u64; 4] {
    const INIT_A: u32 = 0x43B0_D7E5;
    const MULT_A: u32 = 0x931E_8875;
    const INIT_B: u32 = 0x8B51_F9DD;
    const MULT_B: u32 = 0x58F3_8DED;
    const MIX_L: u32 = 0xCA01_F9DD;
    const MIX_R: u32 = 0x4973_F715;

    let words: Vec<u32> = if seed == 0 {
        vec![0]
    } else {
        let mut words = Vec::new();
        let mut value = seed;
        while value > 0 {
            words.push((value & 0xFFFF_FFFF) as u32);
            value >>= 32;
        }
        words
    };
    let mut hash_const = INIT_A;
    let mut hashmix = |value: u32| {
        let mut value = value ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_A);
        value = value.wrapping_mul(hash_const);
        value ^ (value >> 16)
    };
    let mix = |x: u32, y: u32| {
        let result = MIX_L.wrapping_mul(x).wrapping_sub(MIX_R.wrapping_mul(y));
        result ^ (result >> 16)
    };
    let mut pool = [0u32; 4];
    for (i, slot) in pool.iter_mut().enumerate() {
        *slot = hashmix(words.get(i).copied().unwrap_or(0));
    }
    for i_src in 0..4 {
        for i_dst in 0..4 {
            if i_src != i_dst {
                let hashed = hashmix(pool[i_src]);
                pool[i_dst] = mix(pool[i_dst], hashed);
            }
        }
    }
    for word in words.iter().skip(4) {
        for slot in pool.iter_mut() {
            let hashed = hashmix(*word);
            *slot = mix(*slot, hashed);
        }
    }
    let mut hash_const = INIT_B;
    let state: Vec<u32> = (0..8)
        .map(|i| {
            let mut data = pool[i % 4] ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            data = data.wrapping_mul(hash_const);
            data ^ (data >> 16)
        })
        .collect();
    [0, 1, 2, 3].map(|k| u64::from(state[2 * k]) | (u64::from(state[2 * k + 1]) << 32))
}

pub struct NumpyRng {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
}

impl NumpyRng {
    pub fn new(seed: u64) -> Self {
        let words = seed_sequence_state(seed);
        let initstate = (u128::from(words[0]) << 64) | u128::from(words[1]);
        let initseq = (u128::from(words[2]) << 64) | u128::from(words[3]);
        let mut rng = NumpyRng {
            state: 0,
            inc: (initseq << 1) | 1,
            has_uint32: false,
            uinteger: 0,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULTIPLIER).wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let xored = ((self.state >> 64) as u64) ^ (self.state as u64);
        xored.rotate_right((self.state >> 122) as u32)
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let next = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (next >> 32) as u32;
        (next & 0xFFFF_FFFF) as u32
    }

    fn next_double(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    pub fn random(&mut self, size: usize) -> Vec<f64> {
        (0..size).map(|_| self.next_double()).collect()
    }

    pub fn uniform(&mut self, low: f64, high: f64, size: usize) -> Vec<f64> {
        let range = high - low;
        (0..size).map(|_| low + range * self.next_double()).collect()
    }

    fn standard_normal(&mut self) -> f64 {
        loop {
            let mut r = self.next_u64();
            let idx = (r & 0xff) as usize;
            r >>= 8;
            let sign = r & 0x1;
            let rabs = (r >> 1) & 0x000f_ffff_ffff_ffff;
            let mut x = rabs as f64 * WI_DOUBLE[idx];
            if sign & 0x1 == 1 {
                x = -x;
            }
            if rabs < KI_DOUBLE[idx] {
                return x;
            }
            if idx == 0 {
                loop {
                    let xx = -ZIGGURAT_NOR_INV_R * (-self.next_double()).ln_1p();
                    let yy = -(-self.next_double()).ln_1p();
                    if yy + yy > xx * xx {
                        return if (rabs >> 8) & 0x1 == 1 {
                            -(ZIGGURAT_NOR_R + xx)
                        } else {
                            ZIGGURAT_NOR_R + xx
                        };
                    }
                }
            } else if ((FI_DOUBLE[idx - 1] - FI_DOUBLE[idx]) * self.next_double() + FI_DOUBLE[idx])
                < (-0.5 * x * x).exp()
            {
                return x;
            }
        }
    }

    pub fn normal(&mut self, loc: f64, scale: f64, size: usize) -> Vec<f64> {
        (0..size).map(|_| loc + scale * self.standard_normal()).collect()
    }

    pub fn standard_normal_vec(&mut self, size: usize) -> Vec<f64> {
        (0..size).map(|_| self.standard_normal()).collect()
    }

    fn bounded_lemire_u32(&mut self, rng: u32) -> u32 {
        let rng_excl = rng + 1;
        let mut m = u64::from(self.next_u32()) * u64::from(rng_excl);
        let mut leftover = (m & 0xFFFF_FFFF) as u32;
        if leftover < rng_excl {
            let threshold = (u32::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = u64::from(self.next_u32()) * u64::from(rng_excl);
                leftover = (m & 0xFFFF_FFFF) as u32;
            }
        }
        (m >> 32) as u32
    }

    fn bounded_lemire_u64(&mut self, rng: u64) -> u64 {
        let rng_excl = rng + 1;
        let mut m = u128::from(self.next_u64()) * u128::from(rng_excl);
        let mut leftover = m as u64;
        if leftover < rng_excl {
            let threshold = (u64::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = u128::from(self.next_u64()) * u128::from(rng_excl);
                leftover = m as u64;
            }
        }
        (m >> 64) as u64
    }

    fn bounded(&mut self, rng: u64) -> u64 {
        if rng == 0 {
            0
        } else if rng <= 0xFFFF_FFFF {
            if rng == 0xFFFF_FFFF {
                u64::from(self.next_u32())
            } else {
                u64::from(self.bounded_lemire_u32(rng as u32))
            }
        } else if rng == u64::MAX {
            self.next_u64()
        } else {
            self.bounded_lemire_u64(rng)
        }
    }

    pub fn integers(&mut self, low: i64, high: i64, size: usize) -> Vec<i64> {
        let rng = (high - 1 - low) as u64;
        (0..size).map(|_| low + self.bounded(rng) as i64).collect()
    }

    pub fn choice(&mut self, pop_size: i64, size: i64) -> Vec<i64> {
        if pop_size > 10_000 && size > pop_size / 50 {
            let mut idx: Vec<i64> = (0..pop_size).collect();
            let first = (pop_size - size).max(1) as usize;
            for i in (first..pop_size as usize).rev() {
                let j = self.bounded(i as u64) as usize;
                idx.swap(i, j);
            }
            return idx[(pop_size - size) as usize..].to_vec();
        }
        let mut idx = vec![0i64; size as usize];
        let mut mask = (1.2 * size as f64) as u64;
        for shift in [1, 2, 4, 8, 16, 32] {
            mask |= mask >> shift;
        }
        let mut hash_set = vec![u64::MAX; (mask + 1) as usize];
        for j in (pop_size - size)..pop_size {
            let val = self.bounded(j as u64);
            let mut loc = val & mask;
            while hash_set[loc as usize] != u64::MAX && hash_set[loc as usize] != val {
                loc = (loc + 1) & mask;
            }
            let slot = (j - pop_size + size) as usize;
            if hash_set[loc as usize] == u64::MAX {
                hash_set[loc as usize] = val;
                idx[slot] = val as i64;
            } else {
                loc = (j as u64) & mask;
                while hash_set[loc as usize] != u64::MAX {
                    loc = (loc + 1) & mask;
                }
                hash_set[loc as usize] = j as u64;
                idx[slot] = j;
            }
        }
        for i in (1..size as usize).rev() {
            let j = self.bounded(i as u64) as usize;
            idx.swap(i, j);
        }
        idx
    }

    pub fn choice_with_replacement(&mut self, pop_size: i64, size: usize) -> Vec<i64> {
        self.integers(0, pop_size, size)
    }
}
