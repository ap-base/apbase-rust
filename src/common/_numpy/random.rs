//! `np.random.default_rng`: SeedSequence + PCG64 and the bounded-integer
//! algorithms behind `Generator.integers` and `Generator.choice(replace=False)`.
//!
//! Ported from numpy's `bit_generator.pyx`, `pcg64.h`, `distributions.c` and
//! `_generator.pyx`; the same seed yields the same draws as numpy.

const INIT_A: u32 = 0x43B0_D7E5;
const MULT_A: u32 = 0x931E_8875;
const INIT_B: u32 = 0x8B51_F9DD;
const MULT_B: u32 = 0x58F3_8DED;
const MIX_MULT_L: u32 = 0xCA01_F9DD;
const MIX_MULT_R: u32 = 0x4973_F715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

const PCG_MULTIPLIER: u128 = (2_549_297_995_355_413_924u128 << 64) + 4_865_540_595_714_422_341u128;

/// `numpy.random.SeedSequence` restricted to an integer entropy and no spawn key.
#[derive(Clone, Debug)]
pub(crate) struct SeedSequence {
    pool: [u32; POOL_SIZE],
}

impl SeedSequence {
    /// `SeedSequence(entropy)` for a non-negative integer entropy.
    pub(crate) fn new(entropy: u128) -> Self {
        let words = int_to_uint32_words(entropy);
        let mut pool = [0u32; POOL_SIZE];
        let mut hash_const = INIT_A;
        let mut hashmix = |value: u32| -> u32 {
            let mut value = value ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_A);
            value = value.wrapping_mul(hash_const);
            value ^ (value >> XSHIFT)
        };
        for (i, slot) in pool.iter_mut().enumerate() {
            *slot = hashmix(words.get(i).copied().unwrap_or(0));
        }
        for i_src in 0..POOL_SIZE {
            for i_dst in 0..POOL_SIZE {
                if i_src != i_dst {
                    let hashed = hashmix(pool[i_src]);
                    pool[i_dst] = mix(pool[i_dst], hashed);
                }
            }
        }
        for word in words.iter().skip(POOL_SIZE) {
            for slot in pool.iter_mut() {
                let hashed = hashmix(*word);
                *slot = mix(*slot, hashed);
            }
        }
        SeedSequence { pool }
    }

    /// `generate_state(n_words, np.uint64)`.
    pub(crate) fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        let mut hash_const = INIT_B;
        let words: Vec<u32> = (0..n_words * 2)
            .map(|i_dst| {
                let mut data = self.pool[i_dst % POOL_SIZE] ^ hash_const;
                hash_const = hash_const.wrapping_mul(MULT_B);
                data = data.wrapping_mul(hash_const);
                data ^ (data >> XSHIFT)
            })
            .collect();
        words
            .chunks_exact(2)
            .map(|pair| u64::from(pair[0]) | (u64::from(pair[1]) << 32))
            .collect()
    }
}

fn mix(x: u32, y: u32) -> u32 {
    let result = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^ (result >> XSHIFT)
}

/// `_int_to_uint32_array`: little-endian 32-bit words, `[0]` for zero.
fn int_to_uint32_words(mut value: u128) -> Vec<u32> {
    if value == 0 {
        return vec![0];
    }
    let mut words = Vec::new();
    while value > 0 {
        words.push((value & 0xFFFF_FFFF) as u32);
        value >>= 32;
    }
    words
}

/// numpy's `PCG64` bit generator (XSL-RR 128/64), including its 32-bit buffer.
#[derive(Clone, Debug)]
pub(crate) struct Pcg64 {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
}

impl Pcg64 {
    pub(crate) fn from_seed_sequence(seed_sequence: &SeedSequence) -> Self {
        let words = seed_sequence.generate_state_u64(4);
        let initstate = (u128::from(words[0]) << 64) | u128::from(words[1]);
        let initseq = (u128::from(words[2]) << 64) | u128::from(words[3]);
        let mut rng = Pcg64 {
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

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.step();
        let xored = ((self.state >> 64) as u64) ^ (self.state as u64);
        let rotation = (self.state >> 122) as u32;
        xored.rotate_right(rotation)
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let next = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (next >> 32) as u32;
        (next & 0xFFFF_FFFF) as u32
    }

    /// `buffered_bounded_lemire_uint32`: uniform on `[0, rng]`, `rng < u32::MAX`.
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

    /// `bounded_lemire_uint64`: uniform on `[0, rng]`, `rng < u64::MAX`.
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

    /// `random_bounded_uint64(off=0, rng, use_masked=false)`: uniform on `[0, rng]`.
    pub(crate) fn random_bounded_u64(&mut self, rng: u64) -> u64 {
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
}

/// `np.random.Generator` backed by PCG64.
#[derive(Clone, Debug)]
pub(crate) struct Generator {
    bit_generator: Pcg64,
}

impl Generator {
    /// `np.random.default_rng(seed)`; `None` draws 128 bits of OS entropy like numpy.
    pub(crate) fn default_rng(seed: Option<u64>) -> Self {
        let entropy = match seed {
            Some(seed) => u128::from(seed),
            None => {
                let mut bytes = [0u8; 16];
                getrandom::fill(&mut bytes).expect("the operating system entropy source is unavailable");
                u128::from_le_bytes(bytes)
            }
        };
        Generator {
            bit_generator: Pcg64::from_seed_sequence(&SeedSequence::new(entropy)),
        }
    }

    /// `integers(low, high, size=count)` with `dtype=np.int64`, `endpoint=False`.
    pub(crate) fn integers(&mut self, low: i64, high: i64, count: usize) -> Vec<i64> {
        assert!(high > low, "integers requires high > low");
        let rng = (high - 1).wrapping_sub(low) as u64;
        let offset = low as u64;
        let bits = &mut self.bit_generator;
        (0..count)
            .map(|_| {
                let draw = if rng == 0 {
                    0
                } else if rng <= 0xFFFF_FFFF {
                    if rng == 0xFFFF_FFFF {
                        u64::from(bits.next_u32())
                    } else {
                        u64::from(bits.bounded_lemire_u32(rng as u32))
                    }
                } else if rng == u64::MAX {
                    bits.next_u64()
                } else {
                    bits.bounded_lemire_u64(rng)
                };
                offset.wrapping_add(draw) as i64
            })
            .collect()
    }

    /// `choice(pop_size, size, replace=False)` with the default `shuffle=True`.
    pub(crate) fn choice_without_replacement(&mut self, pop_size: i64, size: i64) -> Vec<i64> {
        assert!(
            (0..=pop_size).contains(&size),
            "Cannot take a larger sample than population"
        );
        let bits = &mut self.bit_generator;
        let cutoff = 50;
        if pop_size > 10_000 && size > pop_size / cutoff {
            // Tail shuffle `size` elements.
            let mut idx: Vec<i64> = (0..pop_size).collect();
            shuffle_int(
                bits,
                pop_size as usize,
                (pop_size - size).max(1) as usize,
                &mut idx,
            );
            return idx[(pop_size - size) as usize..].to_vec();
        }

        // Floyd's algorithm with an open-addressing hash set.
        let mut idx = vec![0i64; size as usize];
        let set_size = (1.2 * size as f64) as u64;
        let mask = gen_mask(set_size);
        let mut hash_set = vec![u64::MAX; (mask + 1) as usize];
        for j in (pop_size - size)..pop_size {
            let val = bits.random_bounded_u64(j as u64);
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
        shuffle_int(bits, size as usize, 1, &mut idx);
        idx
    }
}

/// `_shuffle_int`: Fisher-Yates over positions `n-1 .. first`.
fn shuffle_int(bits: &mut Pcg64, n: usize, first: usize, data: &mut [i64]) {
    for i in (first..n).rev() {
        let j = bits.random_bounded_u64(i as u64) as usize;
        data.swap(i, j);
    }
}

/// `_gen_mask`: smallest all-ones bit mask `>= max_val`.
fn gen_mask(max_val: u64) -> u64 {
    let mut mask = max_val;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_words_follow_numpy() {
        assert_eq!(int_to_uint32_words(0), vec![0]);
        assert_eq!(int_to_uint32_words(42), vec![42]);
        assert_eq!(int_to_uint32_words((1u128 << 32) + 5), vec![5, 1]);
    }

    #[test]
    fn gen_mask_is_smallest_all_ones_mask() {
        assert_eq!(gen_mask(0), 0);
        assert_eq!(gen_mask(1), 1);
        assert_eq!(gen_mask(600), 1023);
        assert_eq!(gen_mask(1024), 2047);
    }

    #[test]
    fn choice_returns_distinct_values_in_range() {
        for (pop, size) in [(10, 3), (600, 500), (20_000, 500), (20_000, 300)] {
            let mut generator = Generator::default_rng(Some(7));
            let mut values = generator.choice_without_replacement(pop, size);
            assert_eq!(values.len(), size as usize);
            assert!(values.iter().all(|value| (0..pop).contains(value)));
            values.sort_unstable();
            values.dedup();
            assert_eq!(values.len(), size as usize);
        }
    }

    #[test]
    fn integers_stay_in_range() {
        let mut generator = Generator::default_rng(None);
        assert!(
            generator
                .integers(0, 17, 1000)
                .iter()
                .all(|value| (0..17).contains(value))
        );
        assert!(generator.integers(0, 1, 10).iter().all(|value| *value == 0));
    }
}
