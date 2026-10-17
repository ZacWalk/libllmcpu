// nnc — AVX2 + FMA encoder implementation.
// VEX 2-byte / 3-byte prefix construction plus the specific vector
// instructions used by the JIT kernels (loads, fmadd, broadcast,
// vpmovzxwd / vpmovsxbd widening, horizontal helpers, ...).

#include "emitter_avx2.h"
#include "jit_buffer.h"

#include <cassert>

static inline uint8_t modrm(const uint8_t mod, const uint8_t reg, const uint8_t rm)
{
	return static_cast<uint8_t>((mod << 6) | ((reg & 7) << 3) | (rm & 7));
}

static inline uint8_t sib(const uint8_t scale, const uint8_t index, const uint8_t base)
{
	return static_cast<uint8_t>((scale << 6) | ((index & 7) << 3) | (base & 7));
}

// VEX bit fields are stored *inverted* relative to REX.
static inline uint8_t vex_inv_high(const uint8_t r) { return static_cast<uint8_t>(((r >> 3) & 1) ^ 1); }
static inline uint8_t vex_inv_vvvv(const uint8_t r) { return static_cast<uint8_t>((~r) & 0xF); }

void avx2_emitter::vex2(const uint8_t r, const uint8_t vvvv, const uint8_t l, const uint8_t pp)
{
	const uint8_t b1 = static_cast<uint8_t>(((r & 1) << 7) | ((vvvv & 0xF) << 3) | ((l & 1) << 2) | (pp & 3));
	buf_.emit_u8(0xC5);
	buf_.emit_u8(b1);
}

void avx2_emitter::vex3(const uint8_t r, const uint8_t x, const uint8_t b,
                        const uint8_t map, const uint8_t w, const uint8_t vvvv, const uint8_t l, const uint8_t pp)
{
	const uint8_t b1 = static_cast<uint8_t>(((r & 1) << 7) | ((x & 1) << 6) | ((b & 1) << 5) | (map & 0x1F));
	const uint8_t b2 = static_cast<uint8_t>(((w & 1) << 7) | ((vvvv & 0xF) << 3) | ((l & 1) << 2) | (pp & 3));
	buf_.emit_u8(0xC4);
	buf_.emit_u8(b1);
	buf_.emit_u8(b2);
}

// ----- VEX.NDS.256.0F.WIG 57 /r : VXORPS ymm, ymm, ymm -------------------
void avx2_emitter::vxorps_ymm(ymm dst, ymm a, ymm b)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s = static_cast<uint8_t>(b);
	assert(d < 8 && static_cast<uint8_t>(a) < 8 && s < 8);
	vex2(vex_inv_high(d), vex_inv_vvvv(static_cast<uint8_t>(a)), 1, 0);
	buf_.emit_u8(0x57);
	buf_.emit_u8(modrm(0b11, d, s));
}

// ----- VEX.256.0F.WIG 10 /r : VMOVUPS ymm, m256 --------------------------
void avx2_emitter::vmovups_ymm_load_basex4(ymm dst, gpr base, gpr index)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));
	assert((bs & 7) != 5 && "base = RBP/R13 needs disp8 form; not implemented");

	vex2(vex_inv_high(d), 0xF, 1, 0);
	buf_.emit_u8(0x10);
	buf_.emit_u8(modrm(0b00, d, 0b100));
	buf_.emit_u8(sib(0b10, ix, bs));
}

// ----- VMOVUPS ymm, [base + index*4 + disp8] -----------------------------
void avx2_emitter::vmovups_ymm_load_basex4_disp8(ymm dst, gpr base, gpr index, const int8_t disp)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));

	vex2(vex_inv_high(d), 0xF, 1, 0);
	buf_.emit_u8(0x10);
	buf_.emit_u8(modrm(0b01, d, 0b100));
	buf_.emit_u8(sib(0b10, ix, bs));
	buf_.emit_u8(static_cast<uint8_t>(disp));
}

// ----- VEX.DDS.256.66.0F38.W0 B8 /r : VFMADD231PS ymm, ymm, m256 ---------
void avx2_emitter::vfmadd231ps_ymm_mem_basex4(ymm acc, ymm a, gpr base, gpr index)
{
	const uint8_t d = static_cast<uint8_t>(acc);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && s1 < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));
	assert((bs & 7) != 5);

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     vex_inv_vvvv(s1), 1, 0b01);
	buf_.emit_u8(0xB8);
	buf_.emit_u8(modrm(0b00, d, 0b100));
	buf_.emit_u8(sib(0b10, ix, bs));
}

// ----- VEX.DDS.256.66.0F38.W0 B8 /r : VFMADD231PS ymm, ymm, [base+ix*4+disp8]
void avx2_emitter::vfmadd231ps_ymm_mem_basex4_disp8(ymm acc, ymm a, gpr base, gpr index, const int8_t disp)
{
	const uint8_t d = static_cast<uint8_t>(acc);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && s1 < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     vex_inv_vvvv(s1), 1, 0b01);
	buf_.emit_u8(0xB8);
	buf_.emit_u8(modrm(0b01, d, 0b100));
	buf_.emit_u8(sib(0b10, ix, bs));
	buf_.emit_u8(static_cast<uint8_t>(disp));
}

// ----- VEX.DDS.256.66.0F38.W0 B8 /r : VFMADD231PS ymm, ymm, [base+ix*8] -
void avx2_emitter::vfmadd231ps_ymm_mem_basex8(ymm acc, ymm a, gpr base, gpr index)
{
	const uint8_t d = static_cast<uint8_t>(acc);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && s1 < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));
	assert((bs & 7) != 5);

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     vex_inv_vvvv(s1), 1, 0b01);
	buf_.emit_u8(0xB8);
	buf_.emit_u8(modrm(0b00, d, 0b100));
	buf_.emit_u8(sib(0b11, ix, bs));
}

// ----- VEX.DDS.256.66.0F38.W0 B8 /r : VFMADD231PS ymm, ymm, [base+ix*8+d8]
void avx2_emitter::vfmadd231ps_ymm_mem_basex8_disp8(ymm acc, ymm a, gpr base, gpr index, const int8_t disp)
{
	const uint8_t d = static_cast<uint8_t>(acc);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && s1 < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     vex_inv_vvvv(s1), 1, 0b01);
	buf_.emit_u8(0xB8);
	buf_.emit_u8(modrm(0b01, d, 0b100));
	buf_.emit_u8(sib(0b11, ix, bs));
	buf_.emit_u8(static_cast<uint8_t>(disp));
}

// ----- VEX.DDS.256.66.0F38.W0 B8 /r : VFMADD231PS ymm, ymm, ymm ---------
void avx2_emitter::vfmadd231ps_ymm_ymm_ymm(ymm acc, ymm a, ymm b)
{
	const uint8_t d = static_cast<uint8_t>(acc);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t s2 = static_cast<uint8_t>(b);
	assert(d < 8 && s1 < 8 && s2 < 8);

	vex3(vex_inv_high(d), 1, vex_inv_high(s2),
	     0b00010, 0,
	     vex_inv_vvvv(s1), 1, 0b01);
	buf_.emit_u8(0xB8);
	buf_.emit_u8(modrm(0b11, d, s2));
}

// ----- VEX.256.66.0F38.W0 33 /r : VPMOVZXWD ymm, m128 (SIB index*2) -----
void avx2_emitter::vpmovzxwd_ymm_load_basex2(ymm dst, gpr base, gpr index)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));
	assert((bs & 7) != 5 && "base = RBP/R13 needs disp8 form; not implemented");

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x33);
	buf_.emit_u8(modrm(0b00, d, 0b100)); // SIB follows
	buf_.emit_u8(sib(0b01, ix, bs)); // scale=*2
}

// ----- VPMOVZXWD ymm, [base + index*2 + disp8] --------------------------
void avx2_emitter::vpmovzxwd_ymm_load_basex2_disp8(ymm dst, gpr base, gpr index, const int8_t disp)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x33);
	buf_.emit_u8(modrm(0b01, d, 0b100));
	buf_.emit_u8(sib(0b01, ix, bs));
	buf_.emit_u8(static_cast<uint8_t>(disp));
}

// ----- VPMOVZXWD ymm, [base + index*2 + disp32] -------------------------
void avx2_emitter::vpmovzxwd_ymm_load_basex2_disp32(ymm dst, gpr base, gpr index, const int32_t disp)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x33);
	buf_.emit_u8(modrm(0b10, d, 0b100));
	buf_.emit_u8(sib(0b01, ix, bs));
	buf_.emit_u32(static_cast<uint32_t>(disp));
}

// ----- VEX.256.66.0F38.W0 21 /r : VPMOVSXBD ymm, m64 (SIB index*1) -------
// 8 packed signed bytes in [base + index] -> 8 sign-extended dwords in ymm.
void avx2_emitter::vpmovsxbd_ymm_load_basex1(ymm dst, gpr base, gpr index)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));
	assert((bs & 7) != 5 && "base = RBP/R13 needs disp8 form; not implemented");

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x21);
	buf_.emit_u8(modrm(0b00, d, 0b100)); // SIB follows
	buf_.emit_u8(sib(0b00, ix, bs)); // scale = *1
}

// ----- VPMOVSXBD ymm, [base + index + disp8] -----------------------------
void avx2_emitter::vpmovsxbd_ymm_load_basex1_disp8(ymm dst, gpr base, gpr index, const int8_t disp)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x21);
	buf_.emit_u8(modrm(0b01, d, 0b100));
	buf_.emit_u8(sib(0b00, ix, bs));
	buf_.emit_u8(static_cast<uint8_t>(disp));
}

// ----- VEX.256.66.0F38.W0 31 /r : VPMOVZXBD ymm, m64 (SIB index*1) -------
// 8 packed unsigned bytes in [base + index] -> 8 zero-extended dwords.
void avx2_emitter::vpmovzxbd_ymm_load_basex1(ymm dst, gpr base, gpr index)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));
	assert((bs & 7) != 5 && "base = RBP/R13 needs disp8 form; not implemented");

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x31);
	buf_.emit_u8(modrm(0b00, d, 0b100));
	buf_.emit_u8(sib(0b00, ix, bs));
}

// ----- VPMOVZXBD ymm, [base + index + disp8] -----------------------------
void avx2_emitter::vpmovzxbd_ymm_load_basex1_disp8(ymm dst, gpr base, gpr index, const int8_t disp)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x31);
	buf_.emit_u8(modrm(0b01, d, 0b100));
	buf_.emit_u8(sib(0b00, ix, bs));
	buf_.emit_u8(static_cast<uint8_t>(disp));
}

// ----- VMOVUPS ymm, [base + index*8] ------------------------------------
void avx2_emitter::vmovups_ymm_load_basex8(ymm dst, gpr base, gpr index)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));
	assert((bs & 7) != 5 && "base = RBP/R13 needs disp8 form; not implemented");

	vex2(vex_inv_high(d), 0xF, 1, 0);
	buf_.emit_u8(0x10);
	buf_.emit_u8(modrm(0b00, d, 0b100));
	buf_.emit_u8(sib(0b11, ix, bs));
}

// ----- VMOVUPS ymm, [base + index*8 + disp8] ----------------------------
void avx2_emitter::vmovups_ymm_load_basex8_disp8(ymm dst, gpr base, gpr index, const int8_t disp)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));

	vex2(vex_inv_high(d), 0xF, 1, 0);
	buf_.emit_u8(0x10);
	buf_.emit_u8(modrm(0b01, d, 0b100));
	buf_.emit_u8(sib(0b11, ix, bs));
	buf_.emit_u8(static_cast<uint8_t>(disp));
}

// ----- VEX.NDS.256.66.0F.WIG DB /r : VPAND ymm, ymm, ymm ----------------
void avx2_emitter::vpand_ymm(ymm dst, ymm a, ymm b)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t s2 = static_cast<uint8_t>(b);
	assert(d < 8 && s1 < 8 && s2 < 8);

	vex2(vex_inv_high(d), vex_inv_vvvv(s1), 1, 0b01);
	buf_.emit_u8(0xDB);
	buf_.emit_u8(modrm(0b11, d, s2));
}

// ----- VEX.NDS.256.66.0F.WIG 76 /r : VPCMPEQD ymm, ymm, ymm -------------
void avx2_emitter::vpcmpeqd_ymm(ymm dst, ymm a, ymm b)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t s2 = static_cast<uint8_t>(b);
	assert(d < 8 && s1 < 8 && s2 < 8);

	vex2(vex_inv_high(d), vex_inv_vvvv(s1), 1, 0b01);
	buf_.emit_u8(0x76);
	buf_.emit_u8(modrm(0b11, d, s2));
}

// ----- VEX.NDS.256.66.0F.WIG 72 /2 ib : VPSRLD ymm, ymm, imm8 ----------
// dst encoded in vvvv, src in modrm.rm, /2 in modrm.reg.
void avx2_emitter::vpsrld_ymm_imm8(ymm dst, ymm src, const uint8_t imm)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s = static_cast<uint8_t>(src);
	assert(d < 8 && s < 8);

	vex2(1, vex_inv_vvvv(d), 1, 0b01);
	buf_.emit_u8(0x72);
	buf_.emit_u8(modrm(0b11, 2, s));
	buf_.emit_u8(imm);
}

// ----- VEX.256.66.0F38.W0 18 /r : VBROADCASTSS ymm, m32 -----------------
// Loads one f32 from [base] and broadcasts to all 8 lanes.
void avx2_emitter::vbroadcastss_ymm_load_base(ymm dst, gpr base)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	assert(d < 8 && bs < 8);
	assert((bs & 7) != 4 && "base = RSP/R12 needs SIB; not implemented");
	assert((bs & 7) != 5 && "base = RBP/R13 needs disp8; not implemented");

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x18);
	buf_.emit_u8(modrm(0b00, d, bs));
}

// ----- VEX.256.66.0F38.W0 79 /r : VPBROADCASTW ymm, m16 ----------------
// Broadcasts one 16-bit word to all 16 lanes. With a following
// `vpslld ymm, ymm, 16` this lands a bf16 value in the high half of every
// dword, i.e. broadcasts it as f32.
void avx2_emitter::vpbroadcastw_ymm_load_base(ymm dst, gpr base)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t bs = static_cast<uint8_t>(base);
	assert(d < 8 && bs < 8);
	assert((bs & 7) != 4 && "base = RSP/R12 needs SIB; not implemented");
	assert((bs & 7) != 5 && "base = RBP/R13 needs disp8; not implemented");

	vex3(vex_inv_high(d), 1, vex_inv_high(bs),
	     0b00010, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x79);
	buf_.emit_u8(modrm(0b00, d, bs));
}

// ----- VEX.256.0F.WIG 5B /r : VCVTDQ2PS ymm, ymm ------------------------
// 8 packed signed dwords -> 8 packed single-precision floats.
void avx2_emitter::vcvtdq2ps_ymm_ymm(ymm dst, ymm src)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s = static_cast<uint8_t>(src);
	assert(d < 8 && s < 8);

	vex2(vex_inv_high(d), 0xF, 1, 0); // L=1 (256), pp=none
	buf_.emit_u8(0x5B);
	buf_.emit_u8(modrm(0b11, d, s));
}

// ----- VEX.NDS.256.0F.WIG 59 /r : VMULPS ymm, ymm, ymm ------------------
void avx2_emitter::vmulps_ymm_ymm_ymm(ymm dst, ymm a, ymm b)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t s2 = static_cast<uint8_t>(b);
	assert(d < 8 && s1 < 8 && s2 < 8);

	vex2(vex_inv_high(d), vex_inv_vvvv(s1), 1, 0); // L=1, pp=none
	buf_.emit_u8(0x59);
	buf_.emit_u8(modrm(0b11, d, s2));
}

// ----- VEX.NDS.256.66.0F.WIG 72 /6 ib : VPSLLD ymm, ymm, imm8 -----------
// dst encoded in vvvv, src in modrm.rm, /6 in modrm.reg.
void avx2_emitter::vpslld_ymm_imm8(ymm dst, ymm src, const uint8_t imm)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s = static_cast<uint8_t>(src);
	assert(d < 8 && s < 8);

	vex2(1, vex_inv_vvvv(d), 1, 0b01); // L=1 (256), pp=66
	buf_.emit_u8(0x72);
	buf_.emit_u8(modrm(0b11, 6, s));
	buf_.emit_u8(imm);
}

// ----- VEX.256.66.0F3A.W0 19 /r ib : VEXTRACTF128 xmm, ymm, imm8 ---------
void avx2_emitter::vextractf128_xmm_ymm(ymm dst_xmm, ymm src_ymm, const uint8_t lane)
{
	const uint8_t d = static_cast<uint8_t>(dst_xmm);
	const uint8_t s = static_cast<uint8_t>(src_ymm);
	assert(d < 8 && s < 8);
	assert(lane < 2);

	vex3(vex_inv_high(s), 1, vex_inv_high(d),
	     0b00011, 0,
	     0xF, 1, 0b01);
	buf_.emit_u8(0x19);
	buf_.emit_u8(modrm(0b11, s, d));
	buf_.emit_u8(lane);
}

// ----- VEX.NDS.128.0F.WIG 58 /r : VADDPS xmm, xmm, xmm -------------------
void avx2_emitter::vaddps_xmm(ymm dst, ymm a, ymm b)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t s2 = static_cast<uint8_t>(b);
	assert(d < 8 && s1 < 8 && s2 < 8);

	vex2(vex_inv_high(d), vex_inv_vvvv(s1), 0, 0);
	buf_.emit_u8(0x58);
	buf_.emit_u8(modrm(0b11, d, s2));
}

// ----- VEX.NDS.256.0F.WIG 58 /r : VADDPS ymm, ymm, ymm -------------------
void avx2_emitter::vaddps_ymm(ymm dst, ymm a, ymm b)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t s2 = static_cast<uint8_t>(b);
	assert(d < 8 && s1 < 8 && s2 < 8);

	vex2(vex_inv_high(d), vex_inv_vvvv(s1), 1, 0);
	buf_.emit_u8(0x58);
	buf_.emit_u8(modrm(0b11, d, s2));
}

// ----- VEX.NDS.128.F2.0F.WIG 7C /r : VHADDPS xmm, xmm, xmm ---------------
void avx2_emitter::vhaddps_xmm(ymm dst, ymm a, ymm b)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t s2 = static_cast<uint8_t>(b);
	assert(d < 8 && s1 < 8 && s2 < 8);

	vex2(vex_inv_high(d), vex_inv_vvvv(s1), 0, 0b11);
	buf_.emit_u8(0x7C);
	buf_.emit_u8(modrm(0b11, d, s2));
}

// ----- VEX.NDS.256.F2.0F.WIG 7C /r : VHADDPS ymm, ymm, ymm ---------------
// 256-bit form: hadd is performed per 128-bit lane (does NOT cross lanes).
void avx2_emitter::vhaddps_ymm(ymm dst, ymm a, ymm b)
{
	const uint8_t d = static_cast<uint8_t>(dst);
	const uint8_t s1 = static_cast<uint8_t>(a);
	const uint8_t s2 = static_cast<uint8_t>(b);
	assert(d < 8 && s1 < 8 && s2 < 8);

	vex2(vex_inv_high(d), vex_inv_vvvv(s1), 1, 0b11);
	buf_.emit_u8(0x7C);
	buf_.emit_u8(modrm(0b11, d, s2));
}

// ----- VEX.128.0F.WIG 11 /r : VMOVUPS m128, xmm --------------------------
void avx2_emitter::vmovups_xmm_store_basex4(gpr base, gpr index, ymm src_xmm)
{
	const uint8_t s = static_cast<uint8_t>(src_xmm);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(s < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));
	assert((bs & 7) != 5);

	vex2(vex_inv_high(s), 0xF, 0, 0); // pp = none, L=0
	buf_.emit_u8(0x11);
	buf_.emit_u8(modrm(0b00, s, 0b100));
	buf_.emit_u8(sib(0b10, ix, bs));
}

void avx2_emitter::vmovups_xmm_load_basex4_disp8(ymm dst_xmm, gpr base, gpr index, const int8_t disp)
{
	const uint8_t d = static_cast<uint8_t>(dst_xmm);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(d < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));

	vex2(vex_inv_high(d), 0xF, 0, 0);
	buf_.emit_u8(0x10);
	buf_.emit_u8(modrm(0b01, d, 0b100));
	buf_.emit_u8(sib(0b10, ix, bs));
	buf_.emit_u8(static_cast<uint8_t>(disp));
}

void avx2_emitter::vmovups_xmm_store_basex4_disp8(gpr base, gpr index, const int8_t disp, ymm src_xmm)
{
	const uint8_t s = static_cast<uint8_t>(src_xmm);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(s < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));

	vex2(vex_inv_high(s), 0xF, 0, 0);
	buf_.emit_u8(0x11);
	buf_.emit_u8(modrm(0b01, s, 0b100));
	buf_.emit_u8(sib(0b10, ix, bs));
	buf_.emit_u8(static_cast<uint8_t>(disp));
}

// ----- VEX.128.F3.0F.WIG 11 /r : VMOVSS m32, xmm -------------------------
void avx2_emitter::vmovss_store_basex4(gpr base, gpr index, ymm src_xmm)
{
	const uint8_t s = static_cast<uint8_t>(src_xmm);
	const uint8_t bs = static_cast<uint8_t>(base);
	const uint8_t ix = static_cast<uint8_t>(index);
	assert(s < 8 && bs < 8 && ix < 8);
	assert(ix != static_cast<uint8_t>(gpr::rsp));
	assert((bs & 7) != 5);

	vex2(vex_inv_high(s), 0xF, 0, 0b10); // pp = F3
	buf_.emit_u8(0x11);
	buf_.emit_u8(modrm(0b00, s, 0b100));
	buf_.emit_u8(sib(0b10, ix, bs));
}

// ----- VEX.128.0F.WIG 77 : VZEROUPPER -----------------------------------
void avx2_emitter::vzeroupper()
{
	vex2(1, 0xF, 0, 0);
	buf_.emit_u8(0x77);
}

// ===== register-general forms ============================================

static inline uint8_t reg_id(const ymm r) { return static_cast<uint8_t>(r); }

void avx2_emitter::g_rr(const uint8_t map, const uint8_t pp, const uint8_t l, const uint8_t opc,
                        const uint8_t reg, const uint8_t vvvv, const uint8_t rm)
{
	vex3(vex_inv_high(reg), 1, vex_inv_high(rm), map, 0, vex_inv_vvvv(vvvv), l, pp);
	buf_.emit_u8(opc);
	buf_.emit_u8(modrm(0b11, reg, rm));
}

void avx2_emitter::g_rm(const uint8_t map, const uint8_t pp, const uint8_t l, const uint8_t opc,
                        const uint8_t reg, const uint8_t vvvv, const vmem& m)
{
	const uint8_t bs = static_cast<uint8_t>(m.base);
	const uint8_t ix = static_cast<uint8_t>(m.index);
	const bool has_index = m.index != gpr::rsp;
	assert(m.scale <= 3);
	vex3(vex_inv_high(reg), has_index ? vex_inv_high(ix) : 1, vex_inv_high(bs), map, 0, vex_inv_vvvv(vvvv), l, pp);
	buf_.emit_u8(opc);
	// RBP/R13 as base has no disp-less form; RSP/R12 always need a SIB.
	const uint8_t mod = m.disp == 0 && (bs & 7) != 5 ? 0b00 : m.disp >= -128 && m.disp <= 127 ? 0b01 : 0b10;
	const bool use_sib = has_index || (bs & 7) == 4;
	buf_.emit_u8(modrm(mod, reg, use_sib ? 0b100 : bs));
	if (use_sib) buf_.emit_u8(sib(m.scale, has_index ? ix : 0b100, bs));
	if (mod == 0b01) buf_.emit_u8(static_cast<uint8_t>(m.disp));
	if (mod == 0b10) buf_.emit_u32(static_cast<uint32_t>(m.disp));
}

// VEX.256.0F 57 / 58 / 59 / 5B : VXORPS, VADDPS, VMULPS, VCVTDQ2PS
void avx2_emitter::g_vxorps(const ymm d, const ymm a, const ymm b) { g_rr(1, 0, 1, 0x57, reg_id(d), reg_id(a), reg_id(b)); }
void avx2_emitter::g_vaddps(const ymm d, const ymm a, const ymm b) { g_rr(1, 0, 1, 0x58, reg_id(d), reg_id(a), reg_id(b)); }
void avx2_emitter::g_vaddps_xmm(const ymm d, const ymm a, const ymm b) { g_rr(1, 0, 0, 0x58, reg_id(d), reg_id(a), reg_id(b)); }
void avx2_emitter::g_vmulps(const ymm d, const ymm a, const ymm b) { g_rr(1, 0, 1, 0x59, reg_id(d), reg_id(a), reg_id(b)); }
void avx2_emitter::g_vcvtdq2ps(const ymm d, const ymm s) { g_rr(1, 0, 1, 0x5B, reg_id(d), 0, reg_id(s)); }

// VEX.128.F2.0F 7C : VHADDPS xmm
void avx2_emitter::g_vhaddps_xmm(const ymm d, const ymm a, const ymm b) { g_rr(1, 0b11, 0, 0x7C, reg_id(d), reg_id(a), reg_id(b)); }

// VEX.256.66.0F 72 /6 ib (VPSLLD), /2 ib (VPSRLD): the destination is vvvv.
void avx2_emitter::g_vpslld(const ymm d, const ymm s, const uint8_t imm)
{
	g_rr(1, 0b01, 1, 0x72, 6, reg_id(d), reg_id(s));
	buf_.emit_u8(imm);
}

void avx2_emitter::g_vpsrld(const ymm d, const ymm s, const uint8_t imm)
{
	g_rr(1, 0b01, 1, 0x72, 2, reg_id(d), reg_id(s));
	buf_.emit_u8(imm);
}

// VEX.256.66.0F3A.W0 19 /r ib : VEXTRACTF128 xmm/m128, ymm (source in reg)
void avx2_emitter::g_vextractf128(const ymm d_xmm, const ymm s, const uint8_t lane)
{
	g_rr(3, 0b01, 1, 0x19, reg_id(s), 0, reg_id(d_xmm));
	buf_.emit_u8(lane);
}

// VEX.256.66.0F38.W0 B8 : VFMADD231PS ymm, ymm, m256
void avx2_emitter::g_vfmadd231ps(const ymm acc, const ymm a, const vmem& m) { g_rm(2, 0b01, 1, 0xB8, reg_id(acc), reg_id(a), m); }

// VEX.256.66.0F38 21 / 31 / 79 : VPMOVSXBD, VPMOVZXBD, VPBROADCASTW
void avx2_emitter::g_vpmovsxbd(const ymm d, const vmem& m) { g_rm(2, 0b01, 1, 0x21, reg_id(d), 0, m); }
void avx2_emitter::g_vpmovzxbd(const ymm d, const vmem& m) { g_rm(2, 0b01, 1, 0x31, reg_id(d), 0, m); }
void avx2_emitter::g_vpbroadcastw(const ymm d, const vmem& m) { g_rm(2, 0b01, 1, 0x79, reg_id(d), 0, m); }

// VEX.LIG.F3.0F 11 : VMOVSS m32, xmm ; VEX.128.0F 11 / 10 : VMOVUPS m128
void avx2_emitter::g_vmovss_store(const vmem& m, const ymm s) { g_rm(1, 0b10, 0, 0x11, reg_id(s), 0, m); }
void avx2_emitter::g_vmovups_xmm_store(const vmem& m, const ymm s) { g_rm(1, 0, 0, 0x11, reg_id(s), 0, m); }
void avx2_emitter::g_vmovups_xmm_load(const ymm d, const vmem& m) { g_rm(1, 0, 0, 0x10, reg_id(d), 0, m); }
