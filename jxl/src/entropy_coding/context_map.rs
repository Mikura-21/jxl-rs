// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::HashSet;

use crate::bit_reader::BitReader;
use crate::entropy_coding::decode::*;
use crate::error::Error;

#[derive(Debug)]
pub enum ContextMap {
    AllZero(usize),
    Dense(Vec<u8>),
}

impl ContextMap {
    pub fn len(&self) -> usize {
        match self {
            Self::AllZero(len) => *len,
            Self::Dense(map) => map.len(),
        }
    }

    pub fn last(&self) -> u8 {
        match self {
            Self::AllZero(_) => 0,
            Self::Dense(map) => *map.last().unwrap(),
        }
    }

    pub fn max(&self) -> u8 {
        match self {
            Self::AllZero(_) => 0,
            Self::Dense(map) => *map.iter().max().unwrap(),
        }
    }

    pub fn resize(&mut self, num_contexts: usize) {
        match self {
            Self::AllZero(len) => *len = num_contexts,
            Self::Dense(map) => map.resize(num_contexts, 0),
        }
    }

    pub fn into_dense(self) -> Vec<u8> {
        match self {
            Self::AllZero(len) => vec![0; len],
            Self::Dense(map) => map,
        }
    }
}

fn move_to_front(v: &mut [u8], index: u8) {
    let value = v[index as usize];
    for i in (1..=index as usize).rev() {
        v[i] = v[i - 1];
    }
    v[0] = value;
}

fn inverse_move_to_front(v: &mut [u8]) {
    let mut mtf: [u8; 256] = std::array::from_fn(|x| x as u8);
    for val in v.iter_mut() {
        let index = *val;
        *val = mtf[index as usize];
        if index != 0 {
            move_to_front(&mut mtf, index);
        }
    }
}

fn verify_context_map(ctx_map: &[u8]) -> Result<(), Error> {
    let num_histograms = *ctx_map.iter().max().unwrap() as u32 + 1;
    let distinct_histograms = ctx_map.iter().collect::<HashSet<_>>().len() as u32;
    if distinct_histograms != num_histograms {
        return Err(Error::InvalidContextMapHole(
            num_histograms,
            distinct_histograms,
        ));
    }
    Ok(())
}

pub fn decode_context_map(
    num_contexts: usize,
    br: &mut BitReader,
    lz77_scratch: &mut Vec<u32>,
) -> Result<ContextMap, Error> {
    let is_simple = br.read(1)? != 0;
    if is_simple {
        let bits_per_entry = br.read(2)? as usize;
        let ctx_map: ContextMap = if bits_per_entry != 0 {
            let mut map = Vec::with_capacity(num_contexts);
            for _ in 0..num_contexts {
                map.push(br.read(bits_per_entry)? as u8);
            }
            ContextMap::Dense(map)
        } else {
            ContextMap::AllZero(num_contexts)
        };
        // The hole check is trivial for the all-zero form.
        if let ContextMap::Dense(map) = &ctx_map {
            verify_context_map(map)?;
        }
        Ok(ctx_map)
    } else {
        let use_mtf = br.read(1)? != 0;
        let histograms =
            Histograms::decode(1, br, /*allow_lz77=*/ num_contexts > 2, lz77_scratch)?;
        let mut reader = SymbolReader::new(&histograms, br, None, lz77_scratch)?;

        let mut map: Vec<u8> = (0..num_contexts)
            .map(|_| {
                let mv = reader.read_unsigned(&histograms, br, 0usize);
                if mv > u8::MAX as u32 {
                    Err(Error::InvalidContextMap(mv))
                } else {
                    Ok(mv as u8)
                }
            })
            .collect::<Result<_, _>>()?;
        reader.reclaim_lz77_scratch(lz77_scratch);
        reader.check_final_state(&histograms, br)?;
        if use_mtf {
            inverse_move_to_front(&mut map[..]);
        }
        verify_context_map(&map)?;
        Ok(ContextMap::Dense(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Bits are consumed LSB-first: first `is_simple`, then `bits_per_entry`.
    const SIMPLE_ZERO_BITS: u8 = 0b1; // is_simple=1, bits_per_entry=0
    const SIMPLE_ONE_ENTRY_BITS: u8 = 0b0001_1011; // is_simple=1, bpe=1, entries 1,1,0

    fn decode(bytes: &[u8], num_contexts: usize) -> Result<ContextMap, Error> {
        let mut br = BitReader::new(bytes);
        decode_context_map(num_contexts, &mut br, &mut Vec::new())
    }

    #[test]
    fn simple_zero_bits_map() {
        let map = decode(&[SIMPLE_ZERO_BITS], 1 << 26).unwrap();
        assert!(matches!(map, ContextMap::AllZero(n) if n == 1 << 26));
        assert_eq!((map.max(), map.last()), (0, 0));

        let mut map = ContextMap::AllZero(16);
        map.resize(4096 + 16);
        assert_eq!(map.into_dense(), vec![0u8; 4096 + 16]);
    }

    #[test]
    fn simple_entries_map_and_hole_check() {
        assert_eq!(
            decode(&[SIMPLE_ONE_ENTRY_BITS], 3).unwrap().into_dense(),
            vec![1, 1, 0]
        );
        // Entries 1, 1 leave a hole (one distinct entry for max + 1 = 2).
        let err = decode(&[SIMPLE_ONE_ENTRY_BITS], 2).unwrap_err();
        assert!(matches!(err, Error::InvalidContextMapHole(2, 1)));
    }
}
