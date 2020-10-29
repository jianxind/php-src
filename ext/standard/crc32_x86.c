/*
  +----------------------------------------------------------------------+
  | Copyright (c) The PHP Group                                          |
  +----------------------------------------------------------------------+
  | This source file is subject to version 3.01 of the PHP license,      |
  | that is bundled with this package in the file LICENSE, and is        |
  | available through the world-wide-web at the following url:           |
  | https://www.php.net/license/3_01.txt                                 |
  | If you did not receive a copy of the PHP license and are unable to   |
  | obtain it through the world-wide-web, please send a note to          |
  | license@php.net so we can mail you a copy immediately.               |
  +----------------------------------------------------------------------+
  | Author: Frank Du <frank.du@intel.com>                                |
  +----------------------------------------------------------------------+
  | Compute the crc32 of the buffer. Based on:                           |
  | "Fast CRC Computation for Generic Polynomials Using PCLMULQDQ"       |
  |  V. Gopal, E. Ozturk, et al., 2009, http://intel.ly/2ySEwL0          |
*/

#include "crc32_x86.h"

#if ZEND_INTRIN_SSE4_2_PCLMUL_NATIVE || ZEND_INTRIN_SSE4_2_PCLMUL_RESOLVER
# include <nmmintrin.h>
# include <wmmintrin.h>
#endif
#if ZEND_INTRIN_AVX512_VPCLMULQDQ_RESOLVER
# include <immintrin.h>
#endif

#if ZEND_INTRIN_SSE4_2_PCLMUL_RESOLVER
# include "Zend/zend_cpuinfo.h"
#endif

#if ZEND_INTRIN_SSE4_2_PCLMUL_NATIVE || ZEND_INTRIN_SSE4_2_PCLMUL_RESOLVER

typedef struct _crc32_pclmul_bit_consts {
	uint64_t k1k2[2];
	uint64_t k3k4[2];
	uint64_t k5k6[2];
	uint64_t uPx[2];
} crc32_pclmul_consts;

static const crc32_pclmul_consts crc32_pclmul_consts_maps[X86_CRC32_MAX] = {
	{ /* X86_CRC32, polynomial: 0x04C11DB7 */
		{0x00e6228b11, 0x008833794c}, /* endianness swap */
		{0x00e8a45605, 0x00c5b9cd4c}, /* endianness swap */
		{0x00490d678d, 0x00f200aa66}, /* endianness swap */
		{0x0104d101df, 0x0104c11db7}
	},
	{ /* X86_CRC32B, polynomial: 0x04C11DB7 with reversed ordering */
		{0x0154442bd4, 0x01c6e41596},
		{0x01751997d0, 0x00ccaa009e},
		{0x0163cd6124, 0x01db710640},
		{0x01f7011641, 0x01db710641},
	},
	{ /* X86_CRC32C, polynomial: 0x1EDC6F41 with reversed ordering */
		{0x00740eef02, 0x009e4addf8},
		{0x00f20c0dfe, 0x014cd00bd6},
		{0x00dd45aab8, 0x0000000000},
		{0x00dea713f1, 0x0105ec76f0}
	}
};

static uint8_t pclmul_shuf_mask_table[16] = {
	0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08,
	0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00,
};

/* Folding of 128-bit data chunks */
#define CRC32_FOLDING_BLOCK_SIZE (16)

/* PCLMUL version of non-relfected crc32 */
ZEND_INTRIN_SSE4_2_PCLMUL_FUNC_DECL(size_t crc32_pclmul_batch(uint32_t *crc, const unsigned char *p, size_t nr, const crc32_pclmul_consts *consts));
size_t crc32_pclmul_batch(uint32_t *crc, const unsigned char *p, size_t nr, const crc32_pclmul_consts *consts)
{
	size_t nr_in = nr;
	__m128i x0, x1, x2, k, shuf_mask;

	if (nr < CRC32_FOLDING_BLOCK_SIZE) {
		return 0;
	}

	shuf_mask = _mm_loadu_si128((__m128i *)(pclmul_shuf_mask_table));
	x0 = _mm_cvtsi32_si128(*crc);
	x1 = _mm_loadu_si128((__m128i *)(p + 0x00));
	x0 = _mm_slli_si128(x0, 12);
	x1 = _mm_shuffle_epi8(x1, shuf_mask); /* endianness swap */
	x0 = _mm_xor_si128(x1, x0);
	p += CRC32_FOLDING_BLOCK_SIZE;
	nr -= CRC32_FOLDING_BLOCK_SIZE;

	if (nr >= (CRC32_FOLDING_BLOCK_SIZE * 3)) {
		__m128i x3, x4;

		x1 = _mm_loadu_si128((__m128i *)(p + 0x00));
		x1 = _mm_shuffle_epi8(x1, shuf_mask); /* endianness swap */
		x2 = _mm_loadu_si128((__m128i *)(p + 0x10));
		x2 = _mm_shuffle_epi8(x2, shuf_mask); /* endianness swap */
		x3 = _mm_loadu_si128((__m128i *)(p + 0x20));
		x3 = _mm_shuffle_epi8(x3, shuf_mask); /* endianness swap */
		p += CRC32_FOLDING_BLOCK_SIZE * 3;
		nr -= CRC32_FOLDING_BLOCK_SIZE * 3;

		k = _mm_loadu_si128((__m128i *)consts->k1k2);
		/* parallel folding by 4 */
		while (nr >= (CRC32_FOLDING_BLOCK_SIZE * 4)) {
			__m128i x5, x6, x7, x8, x9, x10, x11;
			x4 = _mm_clmulepi64_si128(x0, k, 0x00);
			x5 = _mm_clmulepi64_si128(x1, k, 0x00);
			x6 = _mm_clmulepi64_si128(x2, k, 0x00);
			x7 = _mm_clmulepi64_si128(x3, k, 0x00);
			x0 = _mm_clmulepi64_si128(x0, k, 0x11);
			x1 = _mm_clmulepi64_si128(x1, k, 0x11);
			x2 = _mm_clmulepi64_si128(x2, k, 0x11);
			x3 = _mm_clmulepi64_si128(x3, k, 0x11);
			x8 = _mm_loadu_si128((__m128i *)(p + 0x00));
			x8 = _mm_shuffle_epi8(x8, shuf_mask); /* endianness swap */
			x9 = _mm_loadu_si128((__m128i *)(p + 0x10));
			x9 = _mm_shuffle_epi8(x9, shuf_mask); /* endianness swap */
			x10 = _mm_loadu_si128((__m128i *)(p + 0x20));
			x10 = _mm_shuffle_epi8(x10, shuf_mask); /* endianness swap */
			x11 = _mm_loadu_si128((__m128i *)(p + 0x30));
			x11 = _mm_shuffle_epi8(x11, shuf_mask); /* endianness swap */
			x0 = _mm_xor_si128(x0, x4);
			x1 = _mm_xor_si128(x1, x5);
			x2 = _mm_xor_si128(x2, x6);
			x3 = _mm_xor_si128(x3, x7);
			x0 = _mm_xor_si128(x0, x8);
			x1 = _mm_xor_si128(x1, x9);
			x2 = _mm_xor_si128(x2, x10);
			x3 = _mm_xor_si128(x3, x11);

			p += CRC32_FOLDING_BLOCK_SIZE * 4;
			nr -= CRC32_FOLDING_BLOCK_SIZE * 4;
		}

		k = _mm_loadu_si128((__m128i *)consts->k3k4);
		/* fold 4 to 1, [x1, x2, x3] -> x0 */
		x4 = _mm_clmulepi64_si128(x0, k, 0x00);
		x0 = _mm_clmulepi64_si128(x0, k, 0x11);
		x0 = _mm_xor_si128(x0, x1);
		x0 = _mm_xor_si128(x0, x4);
		x4 = _mm_clmulepi64_si128(x0, k, 0x00);
		x0 = _mm_clmulepi64_si128(x0, k, 0x11);
		x0 = _mm_xor_si128(x0, x2);
		x0 = _mm_xor_si128(x0, x4);
		x4 = _mm_clmulepi64_si128(x0, k, 0x00);
		x0 = _mm_clmulepi64_si128(x0, k, 0x11);
		x0 = _mm_xor_si128(x0, x3);
		x0 = _mm_xor_si128(x0, x4);
	}

	k = _mm_loadu_si128((__m128i *)consts->k3k4);
	/* folding by 1 */
	while (nr >= CRC32_FOLDING_BLOCK_SIZE) {
		/* load next to x2, fold to x0, x1 */
		x2 = _mm_loadu_si128((__m128i *)(p + 0x00));
		x2 = _mm_shuffle_epi8(x2, shuf_mask); /* endianness swap */
		x1 = _mm_clmulepi64_si128(x0, k, 0x00);
		x0 = _mm_clmulepi64_si128(x0, k, 0x11);
		x0 = _mm_xor_si128(x0, x2);
		x0 = _mm_xor_si128(x0, x1);
		p += CRC32_FOLDING_BLOCK_SIZE;
		nr -= CRC32_FOLDING_BLOCK_SIZE;
	}

	/* reduce 128-bits(final fold) to 96-bits */
	k = _mm_loadu_si128((__m128i*)consts->k5k6);
	x1 = _mm_clmulepi64_si128(x0, k, 0x11);
	x0 = _mm_slli_si128(x0, 8);
	x0 = _mm_srli_si128(x0, 4);
	x0 = _mm_xor_si128(x0, x1);
	/* reduce 96-bits to 64-bits */
	x1 = _mm_clmulepi64_si128(x0, k, 0x01);
	x0 = _mm_xor_si128(x0, x1);

	/* barrett reduction */
	k = _mm_loadu_si128((__m128i*)consts->uPx);
	x1 = _mm_move_epi64(x0);
	x1 = _mm_srli_si128(x1, 4);
	x1 = _mm_clmulepi64_si128(x1, k, 0x00);
	x1 = _mm_srli_si128(x1, 4);
	x1 = _mm_clmulepi64_si128(x1, k, 0x10);
	x0 = _mm_xor_si128(x1, x0);
	*crc =  _mm_extract_epi32(x0, 0);
	return (nr_in - nr); /* the nr processed */
}

/* PCLMUL version of relfected crc32 */
ZEND_INTRIN_SSE4_2_PCLMUL_FUNC_DECL(size_t crc32_pclmul_reflected_batch(uint32_t *crc, const unsigned char *p, size_t nr, const crc32_pclmul_consts *consts));
size_t crc32_pclmul_reflected_batch(uint32_t *crc, const unsigned char *p, size_t nr, const crc32_pclmul_consts *consts)
{
	size_t nr_in = nr;
	__m128i x0, x1, x2, k;

	if (nr < CRC32_FOLDING_BLOCK_SIZE) {
		return 0;
	}

	x0 = _mm_loadu_si128((__m128i *)(p + 0x00));
	x0 = _mm_xor_si128(x0, _mm_cvtsi32_si128(*crc));
	p += CRC32_FOLDING_BLOCK_SIZE;
	nr -= CRC32_FOLDING_BLOCK_SIZE;
	if (nr >= (CRC32_FOLDING_BLOCK_SIZE * 3)) {
		__m128i x3, x4;

		x1 = _mm_loadu_si128((__m128i *)(p + 0x00));
		x2 = _mm_loadu_si128((__m128i *)(p + 0x10));
		x3 = _mm_loadu_si128((__m128i *)(p + 0x20));
		p += CRC32_FOLDING_BLOCK_SIZE * 3;
		nr -= CRC32_FOLDING_BLOCK_SIZE * 3;

		k = _mm_loadu_si128((__m128i *)consts->k1k2);
		/* parallel folding by 4 */
		while (nr >= (CRC32_FOLDING_BLOCK_SIZE * 4)) {
			__m128i x5, x6, x7, x8, x9, x10, x11;
			x4 = _mm_clmulepi64_si128(x0, k, 0x00);
			x5 = _mm_clmulepi64_si128(x1, k, 0x00);
			x6 = _mm_clmulepi64_si128(x2, k, 0x00);
			x7 = _mm_clmulepi64_si128(x3, k, 0x00);
			x0 = _mm_clmulepi64_si128(x0, k, 0x11);
			x1 = _mm_clmulepi64_si128(x1, k, 0x11);
			x2 = _mm_clmulepi64_si128(x2, k, 0x11);
			x3 = _mm_clmulepi64_si128(x3, k, 0x11);
			x8 = _mm_loadu_si128((__m128i *)(p + 0x00));
			x9 = _mm_loadu_si128((__m128i *)(p + 0x10));
			x10 = _mm_loadu_si128((__m128i *)(p + 0x20));
			x11 = _mm_loadu_si128((__m128i *)(p + 0x30));
			x0 = _mm_xor_si128(x0, x4);
			x1 = _mm_xor_si128(x1, x5);
			x2 = _mm_xor_si128(x2, x6);
			x3 = _mm_xor_si128(x3, x7);
			x0 = _mm_xor_si128(x0, x8);
			x1 = _mm_xor_si128(x1, x9);
			x2 = _mm_xor_si128(x2, x10);
			x3 = _mm_xor_si128(x3, x11);

			p += CRC32_FOLDING_BLOCK_SIZE * 4;
			nr -= CRC32_FOLDING_BLOCK_SIZE * 4;
		}

		k = _mm_loadu_si128((__m128i *)consts->k3k4);
		/* fold 4 to 1, [x1, x2, x3] -> x0 */
		x4 = _mm_clmulepi64_si128(x0, k, 0x00);
		x0 = _mm_clmulepi64_si128(x0, k, 0x11);
		x0 = _mm_xor_si128(x0, x1);
		x0 = _mm_xor_si128(x0, x4);
		x4 = _mm_clmulepi64_si128(x0, k, 0x00);
		x0 = _mm_clmulepi64_si128(x0, k, 0x11);
		x0 = _mm_xor_si128(x0, x2);
		x0 = _mm_xor_si128(x0, x4);
		x4 = _mm_clmulepi64_si128(x0, k, 0x00);
		x0 = _mm_clmulepi64_si128(x0, k, 0x11);
		x0 = _mm_xor_si128(x0, x3);
		x0 = _mm_xor_si128(x0, x4);
	}

	k = _mm_loadu_si128((__m128i *)consts->k3k4);
	/* folding by 1 */
	while (nr >= CRC32_FOLDING_BLOCK_SIZE) {
		/* load next to x2, fold to x0, x1 */
		x2 = _mm_loadu_si128((__m128i *)(p + 0x00));
		x1 = _mm_clmulepi64_si128(x0, k, 0x00);
		x0 = _mm_clmulepi64_si128(x0, k, 0x11);
		x0 = _mm_xor_si128(x0, x2);
		x0 = _mm_xor_si128(x0, x1);
		p += CRC32_FOLDING_BLOCK_SIZE;
		nr -= CRC32_FOLDING_BLOCK_SIZE;
	}

	/* reduce 128-bits(final fold) to 96-bits */
	x1 = _mm_clmulepi64_si128(x0, k, 0x10);
	x0 = _mm_srli_si128(x0, 8);
	x0 = _mm_xor_si128(x0, x1);
	/* reduce 96-bits to 64-bits */
	x1 = _mm_shuffle_epi32(x0, 0xfc);
	x0 = _mm_shuffle_epi32(x0, 0xf9);
	k = _mm_loadu_si128((__m128i*)consts->k5k6);
	x1 = _mm_clmulepi64_si128(x1, k, 0x00);
	x0 = _mm_xor_si128(x0, x1);

	/* barrett reduction */
	x1 = _mm_shuffle_epi32(x0, 0xf3);
	x0 = _mm_slli_si128(x0, 4);
	k = _mm_loadu_si128((__m128i*)consts->uPx);
	x1 = _mm_clmulepi64_si128(x1, k, 0x00);
	x1 = _mm_clmulepi64_si128(x1, k, 0x10);
	x0 = _mm_xor_si128(x1, x0);
	*crc =  _mm_extract_epi32(x0, 2);
	return (nr_in - nr); /* the nr processed */
}

# if ZEND_INTRIN_SSE4_2_PCLMUL_NATIVE
size_t crc32_x86_simd_update(X86_CRC32_TYPE type, uint32_t *crc, const unsigned char *p, size_t nr)
# else /* ZEND_INTRIN_SSE4_2_PCLMUL_RESOLVER */
size_t crc32_sse42_pclmul_update(X86_CRC32_TYPE type, uint32_t *crc, const unsigned char *p, size_t nr)
# endif
{
	if (type > X86_CRC32_MAX) {
		return 0;
	}
	const crc32_pclmul_consts *consts = &crc32_pclmul_consts_maps[type];

	switch (type) {
	case X86_CRC32:
		return crc32_pclmul_batch(crc, p, nr, consts);
	case X86_CRC32B:
	case X86_CRC32C:
		return crc32_pclmul_reflected_batch(crc, p, nr, consts);
	default:
		return 0;
	}
}
#endif

# if ZEND_INTRIN_AVX512_VPCLMULQDQ_RESOLVER
typedef struct _crc32_vpclmul_bit_consts {
	uint64_t k1k2[2];
} crc32_vpclmul_consts;

static const crc32_vpclmul_consts crc32_vpclmul_consts_maps[X86_CRC32_MAX] = {
	{ /* X86_CRC32, polynomial: 0x04C11DB7 */
		{0x0088fe2237, 0x00cbcf3bcb} // endianness swap
	},
	{ /* X86_CRC32B, polynomial: 0x04C11DB7 with reversed ordering */
		{0x011542778a, 0x01322d1430}
	},
	{ /* X86_CRC32C, polynomial: 0x1EDC6F41 with reversed ordering */
		{0x00dcb17aa4, 0x00b9e02b86}
	}
};

static uint8_t vpclmul_shuf_mask_table[64] = {
	0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00,
	0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00,
	0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00,
	0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00,
};

/* VPCLMUL version of non-relfected crc32 */
ZEND_INTRIN_AVX512_VPCLMULQDQ_FUNC_DECL(size_t crc32_vpclmul_batch(uint32_t *crc, const unsigned char *p, size_t nr, const crc32_pclmul_consts *consts, const crc32_vpclmul_consts *v_consts));
size_t crc32_vpclmul_batch(uint32_t *crc, const unsigned char *p, size_t nr, const crc32_pclmul_consts *consts, const crc32_vpclmul_consts *v_consts)
{
	size_t nr_in = nr;
	__m128i x0, x1, x2, x3, k, shuf_mask;
	int fold4_exist = 0;

	if (nr < CRC32_FOLDING_BLOCK_SIZE) {
		return 0;
	}

	shuf_mask = _mm_loadu_si128((__m128i *)(pclmul_shuf_mask_table));
	x0 = _mm_cvtsi32_si128(*crc);
	x1 = _mm_loadu_si128((__m128i *)(p + 0x00));
	x0 = _mm_slli_si128(x0, 12);
	x1 = _mm_shuffle_epi8(x1, shuf_mask); // endianness swap
	x0 = _mm_xor_si128(x1, x0);
	p += CRC32_FOLDING_BLOCK_SIZE;
	nr -= CRC32_FOLDING_BLOCK_SIZE;

	if (nr >= (CRC32_FOLDING_BLOCK_SIZE * 3)) {
		fold4_exist = 1;
		x1 = _mm_loadu_si128((__m128i *)(p + 0x00));
		x1 = _mm_shuffle_epi8(x1, shuf_mask); // endianness swap
		x2 = _mm_loadu_si128((__m128i *)(p + 0x10));
		x2 = _mm_shuffle_epi8(x2, shuf_mask); // endianness swap
		x3 = _mm_loadu_si128((__m128i *)(p + 0x20));
		x3 = _mm_shuffle_epi8(x3, shuf_mask); // endianness swap
		p += CRC32_FOLDING_BLOCK_SIZE * 3;
		nr -= CRC32_FOLDING_BLOCK_SIZE * 3;
	}

	if (nr >= (CRC32_FOLDING_BLOCK_SIZE * 12)) {
		__m512i z0, z1, z2, z3, z4, zk, z_shuf;
		__m512i z5, z6, z7, z8, z9, z10, z11;

		z_shuf = _mm512_loadu_si512((__m512i *)(vpclmul_shuf_mask_table));
		z0 = _mm512_loadu_si512((__m512i *)(p - 0x40));
		z0 = _mm512_shuffle_epi8(z0, z_shuf); // endianness swap for each fold lane
		z0 = _mm512_inserti32x4(z0, x0, 0);
		z1 = _mm512_loadu_si512((__m512i *)(p + 0x00));
		z1 = _mm512_shuffle_epi8(z1, z_shuf); // endianness swap for each fold lane
		z2 = _mm512_loadu_si512((__m512i *)(p + 0x40));
		z2 = _mm512_shuffle_epi8(z2, z_shuf); // endianness swap for each fold lane
		z3 = _mm512_loadu_si512((__m512i *)(p + 0x80));
		z3 = _mm512_shuffle_epi8(z3, z_shuf); // endianness swap for each fold lane

		p += CRC32_FOLDING_BLOCK_SIZE * 12;
		nr -= CRC32_FOLDING_BLOCK_SIZE * 12;

		zk = _mm512_set4_epi64(v_consts->k1k2[1], v_consts->k1k2[0], v_consts->k1k2[1], v_consts->k1k2[0]);
		/* parallel folding by 16 */
		while (nr >= (CRC32_FOLDING_BLOCK_SIZE * 16)) {
			z4 = _mm512_clmulepi64_epi128(z0, zk, 0x00);
			z5 = _mm512_clmulepi64_epi128(z1, zk, 0x00);
			z6 = _mm512_clmulepi64_epi128(z2, zk, 0x00);
			z7 = _mm512_clmulepi64_epi128(z3, zk, 0x00);
			z0 = _mm512_clmulepi64_epi128(z0, zk, 0x11);
			z1 = _mm512_clmulepi64_epi128(z1, zk, 0x11);
			z2 = _mm512_clmulepi64_epi128(z2, zk, 0x11);
			z3 = _mm512_clmulepi64_epi128(z3, zk, 0x11);
			z8 = _mm512_loadu_si512((__m512i *)(p + 0x00));
			z8 = _mm512_shuffle_epi8(z8, z_shuf); // endianness swap for each fold lane
			z9 = _mm512_loadu_si512((__m512i *)(p + 0x40));
			z9 = _mm512_shuffle_epi8(z9, z_shuf); // endianness swap for each fold lane
			z10 = _mm512_loadu_si512((__m512i *)(p + 0x80));
			z10 = _mm512_shuffle_epi8(z10, z_shuf); // endianness swap for each fold lane
			z11 = _mm512_loadu_si512((__m512i *)(p + 0xC0));
			z11 = _mm512_shuffle_epi8(z11, z_shuf); // endianness swap for each fold lane
			z0 = _mm512_xor_si512(z0, z4);
			z1 = _mm512_xor_si512(z1, z5);
			z2 = _mm512_xor_si512(z2, z6);
			z3 = _mm512_xor_si512(z3, z7);
			z0 = _mm512_xor_si512(z0, z8);
			z1 = _mm512_xor_si512(z1, z9);
			z2 = _mm512_xor_si512(z2, z10);
			z3 = _mm512_xor_si512(z3, z11);

			p += CRC32_FOLDING_BLOCK_SIZE * 16;
			nr -= CRC32_FOLDING_BLOCK_SIZE * 16;
		}

		/* fold 16 to 4, [z1, z2, z3] -> z0 */
		zk = _mm512_set4_epi64(consts->k1k2[1], consts->k1k2[0], consts->k1k2[1], consts->k1k2[0]);
		z4 = _mm512_clmulepi64_epi128(z0, zk, 0x00);
		z0 = _mm512_clmulepi64_epi128(z0, zk, 0x11);
		z0 = _mm512_xor_si512(z0, z1);
		z0 = _mm512_xor_si512(z0, z4);
		z4 = _mm512_clmulepi64_epi128(z0, zk, 0x00);
		z0 = _mm512_clmulepi64_epi128(z0, zk, 0x11);
		z0 = _mm512_xor_si512(z0, z2);
		z0 = _mm512_xor_si512(z0, z4);
		z4 = _mm512_clmulepi64_epi128(z0, zk, 0x00);
		z0 = _mm512_clmulepi64_epi128(z0, zk, 0x11);
		z0 = _mm512_xor_si512(z0, z3);
		z0 = _mm512_xor_si512(z0, z4);

		x0 = _mm512_extracti32x4_epi32(z0, 0);
		x1 = _mm512_extracti32x4_epi32(z0, 1);
		x2 = _mm512_extracti32x4_epi32(z0, 2);
		x3 = _mm512_extracti32x4_epi32(z0, 3);
	}

	k = _mm_loadu_si128((__m128i *)consts->k1k2);
	/* parallel folding by 4 */
	while (nr >= (CRC32_FOLDING_BLOCK_SIZE * 4)) {
		__m128i x4, x5, x6, x7, x8, x9, x10, x11;
		x4 = _mm_clmulepi64_si128(x0, k, 0x00);
		x5 = _mm_clmulepi64_si128(x1, k, 0x00);
		x6 = _mm_clmulepi64_si128(x2, k, 0x00);
		x7 = _mm_clmulepi64_si128(x3, k, 0x00);
		x0 = _mm_clmulepi64_si128(x0, k, 0x11);
		x1 = _mm_clmulepi64_si128(x1, k, 0x11);
		x2 = _mm_clmulepi64_si128(x2, k, 0x11);
		x3 = _mm_clmulepi64_si128(x3, k, 0x11);
		x8 = _mm_loadu_si128((__m128i *)(p + 0x00));
		x8 = _mm_shuffle_epi8(x8, shuf_mask); // endianness swap
		x9 = _mm_loadu_si128((__m128i *)(p + 0x10));
		x9 = _mm_shuffle_epi8(x9, shuf_mask); // endianness swap
		x10 = _mm_loadu_si128((__m128i *)(p + 0x20));
		x10 = _mm_shuffle_epi8(x10, shuf_mask); // endianness swap
		x11 = _mm_loadu_si128((__m128i *)(p + 0x30));
		x11 = _mm_shuffle_epi8(x11, shuf_mask); // endianness swap
		x0 = _mm_xor_si128(x0, x4);
		x1 = _mm_xor_si128(x1, x5);
		x2 = _mm_xor_si128(x2, x6);
		x3 = _mm_xor_si128(x3, x7);
		x0 = _mm_xor_si128(x0, x8);
		x1 = _mm_xor_si128(x1, x9);
		x2 = _mm_xor_si128(x2, x10);
		x3 = _mm_xor_si128(x3, x11);

		p += CRC32_FOLDING_BLOCK_SIZE * 4;
		nr -= CRC32_FOLDING_BLOCK_SIZE * 4;
	}

	k = _mm_loadu_si128((__m128i *)consts->k3k4);
	if (fold4_exist) {
		/* fold 4 to 1, [x1, x2, x3] -> x0 */
		__m128i x4 = _mm_clmulepi64_si128(x0, k, 0x00);
		x0 = _mm_clmulepi64_si128(x0, k, 0x11);
		x0 = _mm_xor_si128(x0, x1);
		x0 = _mm_xor_si128(x0, x4);
		x4 = _mm_clmulepi64_si128(x0, k, 0x00);
		x0 = _mm_clmulepi64_si128(x0, k, 0x11);
		x0 = _mm_xor_si128(x0, x2);
		x0 = _mm_xor_si128(x0, x4);
		x4 = _mm_clmulepi64_si128(x0, k, 0x00);
		x0 = _mm_clmulepi64_si128(x0, k, 0x11);
		x0 = _mm_xor_si128(x0, x3);
		x0 = _mm_xor_si128(x0, x4);
	}
	/* folding by 1 */
	while (nr >= CRC32_FOLDING_BLOCK_SIZE) {
		/* load next to x2, fold to x0, x1 */
		x2 = _mm_loadu_si128((__m128i *)(p + 0x00));
		x2 = _mm_shuffle_epi8(x2, shuf_mask); // endianness swap
		x1 = _mm_clmulepi64_si128(x0, k, 0x00);
		x0 = _mm_clmulepi64_si128(x0, k, 0x11);
		x0 = _mm_xor_si128(x0, x2);
		x0 = _mm_xor_si128(x0, x1);
		p += CRC32_FOLDING_BLOCK_SIZE;
		nr -= CRC32_FOLDING_BLOCK_SIZE;
	}

	/* reduce 128-bits(final fold) to 96-bits */
	k = _mm_loadu_si128((__m128i *)consts->k5k6);
	x1 = _mm_clmulepi64_si128(x0, k, 0x11);
	x0 = _mm_slli_si128(x0, 8);
	x0 = _mm_srli_si128(x0, 4);
	x0 = _mm_xor_si128(x0, x1);
	/* reduce 96-bits to 64-bits */
	x1 = _mm_clmulepi64_si128(x0, k, 0x01);
	x0 = _mm_xor_si128(x0, x1);

	/* barrett reduction */
	k = _mm_loadu_si128((__m128i *)consts->uPx);
	x1 = _mm_loadu_si64(&x0);
	x1 = _mm_srli_si128(x1, 4);
	x1 = _mm_clmulepi64_si128(x1, k, 0x00);
	x1 = _mm_srli_si128(x1, 4);
	x1 = _mm_clmulepi64_si128(x1, k, 0x10);
	x0 = _mm_xor_si128(x1, x0);
	*crc = _mm_extract_epi32(x0, 0);
	return (nr_in - nr); /* the nr handled */
}

/* VPCLMUL version of relfected crc32 */
ZEND_INTRIN_AVX512_VPCLMULQDQ_FUNC_DECL(size_t crc32_vpclmul_reflected_batch(uint32_t *crc, const unsigned char *p, size_t nr, const crc32_pclmul_consts *consts, const crc32_vpclmul_consts *v_consts));
size_t crc32_vpclmul_reflected_batch(uint32_t *crc, const unsigned char *p, size_t nr, const crc32_pclmul_consts *consts, const crc32_vpclmul_consts *v_consts)
{
	size_t nr_in = nr;
	__m128i x0, x1, x2, x3, k;
	int fold4_exist = 0;

	if (nr < CRC32_FOLDING_BLOCK_SIZE) {
		return 0;
	}

	x0 = _mm_loadu_si128((__m128i *)(p + 0x00));
	x0 = _mm_xor_si128(x0, _mm_cvtsi32_si128(*crc));
	p += CRC32_FOLDING_BLOCK_SIZE;
	nr -= CRC32_FOLDING_BLOCK_SIZE;

	if (nr >= (CRC32_FOLDING_BLOCK_SIZE * 3)) {
		fold4_exist = 1;
		x1 = _mm_loadu_si128((__m128i *)(p + 0x00));
		x2 = _mm_loadu_si128((__m128i *)(p + 0x10));
		x3 = _mm_loadu_si128((__m128i *)(p + 0x20));
		p += CRC32_FOLDING_BLOCK_SIZE * 3;
		nr -= CRC32_FOLDING_BLOCK_SIZE * 3;
	}

	if (nr >= (CRC32_FOLDING_BLOCK_SIZE * 12)) {
		__m512i z0, z1, z2, z3, z4, zk;
		__m512i z5, z6, z7, z8, z9, z10, z11;

		z0 = _mm512_loadu_si512((__m512i *)(p - 0x40));
		z0 = _mm512_inserti32x4(z0, x0, 0);
		z1 = _mm512_loadu_si512((__m512i *)(p + 0x00));
		z2 = _mm512_loadu_si512((__m512i *)(p + 0x40));
		z3 = _mm512_loadu_si512((__m512i *)(p + 0x80));
		p += CRC32_FOLDING_BLOCK_SIZE * 12;
		nr -= CRC32_FOLDING_BLOCK_SIZE * 12;

		zk = _mm512_set4_epi64(v_consts->k1k2[1], v_consts->k1k2[0], v_consts->k1k2[1], v_consts->k1k2[0]);

		/* parallel folding by 16 */
		while (nr >= (CRC32_FOLDING_BLOCK_SIZE * 16)) {
			z4 = _mm512_clmulepi64_epi128(z0, zk, 0x00);
			z5 = _mm512_clmulepi64_epi128(z1, zk, 0x00);
			z6 = _mm512_clmulepi64_epi128(z2, zk, 0x00);
			z7 = _mm512_clmulepi64_epi128(z3, zk, 0x00);
			z0 = _mm512_clmulepi64_epi128(z0, zk, 0x11);
			z1 = _mm512_clmulepi64_epi128(z1, zk, 0x11);
			z2 = _mm512_clmulepi64_epi128(z2, zk, 0x11);
			z3 = _mm512_clmulepi64_epi128(z3, zk, 0x11);
			z8 = _mm512_loadu_si512((__m512i *)(p + 0x00));
			z9 = _mm512_loadu_si512((__m512i *)(p + 0x40));
			z10 = _mm512_loadu_si512((__m512i *)(p + 0x80));
			z11 = _mm512_loadu_si512((__m512i *)(p + 0xC0));
			z0 = _mm512_xor_si512(z0, z4);
			z1 = _mm512_xor_si512(z1, z5);
			z2 = _mm512_xor_si512(z2, z6);
			z3 = _mm512_xor_si512(z3, z7);
			z0 = _mm512_xor_si512(z0, z8);
			z1 = _mm512_xor_si512(z1, z9);
			z2 = _mm512_xor_si512(z2, z10);
			z3 = _mm512_xor_si512(z3, z11);

			p += CRC32_FOLDING_BLOCK_SIZE * 16;
			nr -= CRC32_FOLDING_BLOCK_SIZE * 16;
		}

		/* fold 16 to 4, [z1, z2, z3] -> z0	*/
		zk = _mm512_set4_epi64(consts->k1k2[1], consts->k1k2[0], consts->k1k2[1], consts->k1k2[0]);
		z4 = _mm512_clmulepi64_epi128(z0, zk, 0x00);
		z0 = _mm512_clmulepi64_epi128(z0, zk, 0x11);
		z0 = _mm512_xor_si512(z0, z1);
		z0 = _mm512_xor_si512(z0, z4);
		z4 = _mm512_clmulepi64_epi128(z0, zk, 0x00);
		z0 = _mm512_clmulepi64_epi128(z0, zk, 0x11);
		z0 = _mm512_xor_si512(z0, z2);
		z0 = _mm512_xor_si512(z0, z4);
		z4 = _mm512_clmulepi64_epi128(z0, zk, 0x00);
		z0 = _mm512_clmulepi64_epi128(z0, zk, 0x11);
		z0 = _mm512_xor_si512(z0, z3);
		z0 = _mm512_xor_si512(z0, z4);

		x0 = _mm512_extracti32x4_epi32(z0, 0);
		x1 = _mm512_extracti32x4_epi32(z0, 1);
		x2 = _mm512_extracti32x4_epi32(z0, 2);
		x3 = _mm512_extracti32x4_epi32(z0, 3);
	}

	k = _mm_loadu_si128((__m128i *)consts->k1k2);
	/* parallel folding by 4 */
	while (nr >= (CRC32_FOLDING_BLOCK_SIZE * 4)) {
		__m128i x4, x5, x6, x7, x8, x9, x10, x11;
		x4 = _mm_clmulepi64_si128(x0, k, 0x00);
		x5 = _mm_clmulepi64_si128(x1, k, 0x00);
		x6 = _mm_clmulepi64_si128(x2, k, 0x00);
		x7 = _mm_clmulepi64_si128(x3, k, 0x00);
		x0 = _mm_clmulepi64_si128(x0, k, 0x11);
		x1 = _mm_clmulepi64_si128(x1, k, 0x11);
		x2 = _mm_clmulepi64_si128(x2, k, 0x11);
		x3 = _mm_clmulepi64_si128(x3, k, 0x11);
		x8 = _mm_loadu_si128((__m128i *)(p + 0x00));
		x9 = _mm_loadu_si128((__m128i *)(p + 0x10));
		x10 = _mm_loadu_si128((__m128i *)(p + 0x20));
		x11 = _mm_loadu_si128((__m128i *)(p + 0x30));
		x0 = _mm_xor_si128(x0, x4);
		x1 = _mm_xor_si128(x1, x5);
		x2 = _mm_xor_si128(x2, x6);
		x3 = _mm_xor_si128(x3, x7);
		x0 = _mm_xor_si128(x0, x8);
		x1 = _mm_xor_si128(x1, x9);
		x2 = _mm_xor_si128(x2, x10);
		x3 = _mm_xor_si128(x3, x11);

		p += CRC32_FOLDING_BLOCK_SIZE * 4;
		nr -= CRC32_FOLDING_BLOCK_SIZE * 4;
	}

	k = _mm_loadu_si128((__m128i *)consts->k3k4);
	/* fold 4 to 1, [x1, x2, x3] -> x0 */
	if (fold4_exist) {
		__m128i x4 = _mm_clmulepi64_si128(x0, k, 0x00);
		x0 = _mm_clmulepi64_si128(x0, k, 0x11);
		x0 = _mm_xor_si128(x0, x1);
		x0 = _mm_xor_si128(x0, x4);
		x4 = _mm_clmulepi64_si128(x0, k, 0x00);
		x0 = _mm_clmulepi64_si128(x0, k, 0x11);
		x0 = _mm_xor_si128(x0, x2);
		x0 = _mm_xor_si128(x0, x4);
		x4 = _mm_clmulepi64_si128(x0, k, 0x00);
		x0 = _mm_clmulepi64_si128(x0, k, 0x11);
		x0 = _mm_xor_si128(x0, x3);
		x0 = _mm_xor_si128(x0, x4);
	}
	/* folding by 1 */
	while (nr >= CRC32_FOLDING_BLOCK_SIZE) {
		/* load next to x2, fold to x0, x1 */
		x2 = _mm_loadu_si128((__m128i *)(p + 0x00));
		x1 = _mm_clmulepi64_si128(x0, k, 0x00);
		x0 = _mm_clmulepi64_si128(x0, k, 0x11);
		x0 = _mm_xor_si128(x0, x2);
		x0 = _mm_xor_si128(x0, x1);
		p += CRC32_FOLDING_BLOCK_SIZE;
		nr -= CRC32_FOLDING_BLOCK_SIZE;
	}

	/* reduce 128-bits(final fold) to 96-bits */
	x1 = _mm_clmulepi64_si128(x0, k, 0x10);
	x0 = _mm_srli_si128(x0, 8);
	x0 = _mm_xor_si128(x0, x1);
	/* reduce 96-bits to 64-bits */
	x1 = _mm_shuffle_epi32(x0, 0xfc);
	x0 = _mm_shuffle_epi32(x0, 0xf9);
	k = _mm_loadu_si128((__m128i *)consts->k5k6);
	x1 = _mm_clmulepi64_si128(x1, k, 0x00);
	x0 = _mm_xor_si128(x0, x1);

	/* barrett reduction */
	x1 = _mm_shuffle_epi32(x0, 0xf3);
	x0 = _mm_slli_si128(x0, 4);
	k = _mm_loadu_si128((__m128i *)consts->uPx);
	x1 = _mm_clmulepi64_si128(x1, k, 0x00);
	x1 = _mm_clmulepi64_si128(x1, k, 0x10);
	x0 = _mm_xor_si128(x1, x0);
	*crc = _mm_extract_epi32(x0, 2);
	return (nr_in - nr); /* the nr handled */
}

size_t crc32_vpclmul_update(X86_CRC32_TYPE type, uint32_t *crc, const unsigned char *p, size_t nr)
{
	if (type > X86_CRC32_MAX) {
		return 0;
	}
	const crc32_pclmul_consts *consts = &crc32_pclmul_consts_maps[type];
	const crc32_vpclmul_consts *v_consts = &crc32_vpclmul_consts_maps[type];

	switch (type) {
	case X86_CRC32:
		return crc32_vpclmul_batch(crc, p, nr, consts, v_consts);
	case X86_CRC32B:
	case X86_CRC32C:
		return crc32_vpclmul_reflected_batch(crc, p, nr, consts, v_consts);
	default:
		return 0;
	}
}
# endif

#if ZEND_INTRIN_SSE4_2_PCLMUL_RESOLVER
static size_t crc32_x86_simd_update_default(X86_CRC32_TYPE type, uint32_t *crc, const unsigned char *p, size_t nr)
{
	return 0;
}

# if ZEND_INTRIN_SSE4_2_PCLMUL_FUNC_PROTO
size_t crc32_x86_simd_update(X86_CRC32_TYPE type, uint32_t *crc, const unsigned char *p, size_t nr) __attribute__((ifunc("resolve_crc32_x86_simd_update")));

typedef size_t (*crc32_x86_simd_func_t)(X86_CRC32_TYPE type, uint32_t *crc, const unsigned char *p, size_t nr);

ZEND_NO_SANITIZE_ADDRESS
ZEND_ATTRIBUTE_UNUSED /* clang mistakenly warns about this */
static crc32_x86_simd_func_t resolve_crc32_x86_simd_update() {
#  if ZEND_INTRIN_AVX512_VPCLMULQDQ_FUNC_PROTO
	if (zend_cpu_supports_avx512_vpclmulqdq()) {
		return crc32_vpclmul_update;
	}
#  endif
	if (zend_cpu_supports_sse42() && zend_cpu_supports_pclmul()) {
		return crc32_sse42_pclmul_update;
	}
	return crc32_x86_simd_update_default;
}
# else /* ZEND_INTRIN_SSE4_2_PCLMUL_FUNC_PTR */
static size_t (*crc32_x86_simd_ptr)(X86_CRC32_TYPE type, uint32_t *crc, const unsigned char *p, size_t nr) = crc32_x86_simd_update_default;

size_t crc32_x86_simd_update(X86_CRC32_TYPE type, uint32_t *crc, const unsigned char *p, size_t nr) {
	return crc32_x86_simd_ptr(type, crc, p, nr);
}

/* {{{ PHP_MINIT_FUNCTION */
PHP_MINIT_FUNCTION(crc32_x86_intrin)
{
#  if ZEND_INTRIN_AVX512_VPCLMULQDQ_FUNC_PTR
	if (zend_cpu_supports_avx512_vpclmulqdq()) {
		crc32_x86_simd_ptr = crc32_vpclmul_update;
	} else
#  endif
	if (zend_cpu_supports_sse42() && zend_cpu_supports_pclmul()) {
		crc32_x86_simd_ptr = crc32_sse42_pclmul_update;
	}
	return SUCCESS;
}
/* }}} */
# endif
#endif
