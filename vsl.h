#ifndef VSL_H
#define VSL_H

// TODO: Write checks for availability of intrinsics.
// For now, we just assume this is valid.
#include <immintrin.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

// TODO: Remove in the future (just for testing)
#include <stdio.h>

#define VSLAPI static

#define VSL_NOT_IMPLEMENTED(msg) assert(!(msg))

// |  B  |  A  |
// | o o | o o |
// c3, c2 - Picking from second vector (B).
// c1, c0 - Picking from first vector (A).
// If ci has value 2, that means that we are picking 2nd 32bit float from corresponding
// vector.
#define VSL_MM_SHUFFLE_MASK(c3, c2, c1, c0) (((c3) << 6) + ((c2) << 4) + ((c1) << 2) + (c0))

typedef union {
	struct {
		float v0, v1, v2, v3;
	};
	struct {
		float x, y, z, w;
	};
	__m128 vec;
	float arr[4];
} vsl_V4f;

typedef union {
	struct {
		uint32_t v0, v1, v2, v3;
	};
	struct {
		uint32_t x, y, z, w;
	};
	__m128i vec;
	uint32_t arr[4];
} vsl_V4i;

typedef union {
	struct {
		vsl_V4f c0, c1, c2, c3;
	};
	vsl_V4f cols[4];
} vsl_M4x4f;

// TODO: Realize minimal copy and minimal inconvenience.
// Passing things by reference, as well as passing vectors/matrices
// that represent result, instead of returning it by copy.
// Not doing this yet since it is not computational core.

// VECTOR
VSLAPI inline vsl_V4f vsl_v4f_add(vsl_V4f v, vsl_V4f w);
VSLAPI inline vsl_V4f vsl_v4f_sub(vsl_V4f v, vsl_V4f w);
VSLAPI inline vsl_V4f vsl_v4f_mul(vsl_V4f v, vsl_V4f w);
VSLAPI inline vsl_M4x4f vsl_v4f_tensor_mul(vsl_V4f v, vsl_V4f w);
VSLAPI inline vsl_V4f vsl_v4f_div(vsl_V4f v, vsl_V4f w);
VSLAPI inline vsl_V4i vsl_v4i_add(vsl_V4i v, vsl_V4i w);
VSLAPI inline vsl_V4i vsl_v4i_sub(vsl_V4i v, vsl_V4i w);
VSLAPI inline vsl_V4f vsl_v4f_scale(vsl_V4f v, float s);
VSLAPI inline vsl_V4f vsl_v4f_inv(vsl_V4f v);
VSLAPI inline vsl_V4f vsl_v4f_unit(vsl_V4f v);
VSLAPI inline float	  vsl_v4f_dot(vsl_V4f v, vsl_V4f w);
VSLAPI inline vsl_V4f vsl_v4f_cross(vsl_V4f v, vsl_V4f w);
VSLAPI inline vsl_V4f vsl_v4f_sq(vsl_V4f v);
VSLAPI inline float	  vsl_v4f_sum(vsl_V4f v);
VSLAPI inline float	  vsl_v4f_lensq(vsl_V4f v);
VSLAPI inline float	  vsl_v4f_len(vsl_V4f v);

VSLAPI inline void vsl_v4f_add_mut(vsl_V4f *v, vsl_V4f w);
VSLAPI inline void vsl_v4f_sub_mut(vsl_V4f *v, vsl_V4f w);
VSLAPI inline void vsl_v4f_mul_mut(vsl_V4f *v, vsl_V4f w);
VSLAPI inline void vsl_v4f_div_mut(vsl_V4f *v, vsl_V4f w);
VSLAPI inline void vsl_v4i_add_mut(vsl_V4i *v, vsl_V4i w);
VSLAPI inline void vsl_v4i_sub_mut(vsl_V4i *v, vsl_V4i w);
VSLAPI inline void vsl_v4f_scale_mut(vsl_V4f *v, float s);
VSLAPI inline void vsl_v4f_inv_mut(vsl_V4f *v);
VSLAPI inline void vsl_v4f_unit_mut(vsl_V4f *v);
VSLAPI inline void vsl_v4f_sq_mut(vsl_V4f *v);

// MATRIX
VSLAPI inline vsl_M4x4f vsl_m4x4f_add(vsl_M4x4f A, vsl_M4x4f B);
VSLAPI inline vsl_M4x4f vsl_m4x4f_sub(vsl_M4x4f A, vsl_M4x4f B);
VSLAPI inline vsl_M4x4f vsl_m4x4f_mul(vsl_M4x4f A, vsl_M4x4f B);
VSLAPI inline vsl_M4x4f vsl_m4x4f_scale(vsl_M4x4f A, float s);
VSLAPI inline vsl_M4x4f vsl_m4x4f_transpose(vsl_M4x4f A);
VSLAPI inline vsl_M4x4f vsl_m4x4f_inv(vsl_M4x4f A);
VSLAPI inline vsl_V4f vsl_m4x4f_map(vsl_V4f v, vsl_M4x4f A);
VSLAPI inline float vsl_m4x4f_det(vsl_M4x4f A);
VSLAPI inline float vsl_m4x4f_sum(vsl_M4x4f A);

VSLAPI inline void vsl_m4x4f_add_mut(vsl_M4x4f *A, vsl_M4x4f B);
VSLAPI inline void vsl_m4x4f_sub_mut(vsl_M4x4f *A, vsl_M4x4f B);
VSLAPI inline void vsl_m4x4f_mul_mut(vsl_M4x4f *A, vsl_M4x4f B);
VSLAPI inline void vsl_m4x4f_scale_mut(vsl_M4x4f *A, float s);
VSLAPI inline void vsl_m4x4f_transpose_mut(vsl_M4x4f *A);
VSLAPI inline void vsl_m4x4f_inv_mut(vsl_M4x4f *A);
VSLAPI inline void vsl_m4x4f_map_mut(vsl_V4f *v, vsl_M4x4f A);

// TEST
VSLAPI void vsl_v4f_print(vsl_V4f v);
VSLAPI void vsl_v4i_print(vsl_V4i v);
VSLAPI void vsl_M4x4f_print(vsl_M4x4f A);

#endif // VSL_H

#ifdef VSL_IMPLEMENTATION

VSLAPI inline vsl_V4f vsl_v4f_add(vsl_V4f v, vsl_V4f w) {
	return (vsl_V4f)_mm_add_ps(v.vec, w.vec);
}

VSLAPI inline vsl_V4f vsl_v4f_sub(vsl_V4f v, vsl_V4f w) {
	return (vsl_V4f)_mm_sub_ps(v.vec, w.vec);
}

VSLAPI inline vsl_V4f vsl_v4f_mul(vsl_V4f v, vsl_V4f w) {
	return (vsl_V4f)_mm_mul_ps(v.vec, w.vec);
}

VSLAPI inline vsl_M4x4f vsl_v4f_tensor_mul(vsl_V4f v, vsl_V4f w) {
	vsl_M4x4f M = {0};
	
	M.c0 = (vsl_V4f)_mm_mul_ps(v.vec, _mm_shuffle_ps(w.vec, w.vec, VSL_MM_SHUFFLE_MASK(0, 0, 0, 0)));
	M.c1 = (vsl_V4f)_mm_mul_ps(v.vec, _mm_shuffle_ps(w.vec, w.vec, VSL_MM_SHUFFLE_MASK(1, 1, 1, 1)));
	M.c2 = (vsl_V4f)_mm_mul_ps(v.vec, _mm_shuffle_ps(w.vec, w.vec, VSL_MM_SHUFFLE_MASK(2, 2, 2, 2)));
	M.c3 = (vsl_V4f)_mm_mul_ps(v.vec, _mm_shuffle_ps(w.vec, w.vec, VSL_MM_SHUFFLE_MASK(3, 3, 3, 3)));

	return M;
}

VSLAPI inline vsl_V4f vsl_v4f_div(vsl_V4f v, vsl_V4f w) {
	return (vsl_V4f)_mm_div_ps(v.vec, w.vec);
}

VSLAPI inline vsl_V4i vsl_v4i_add(vsl_V4i v, vsl_V4i w) {
	return (vsl_V4i)_mm_add_epi32(v.vec, w.vec);
}

VSLAPI inline vsl_V4i vsl_v4i_sub(vsl_V4i v, vsl_V4i w) {
	return (vsl_V4i)_mm_sub_epi32(v.vec, w.vec);
}

VSLAPI inline vsl_V4f vsl_v4f_scale(vsl_V4f v, float s) {
	vsl_V4f u = {{s, s, s, s}};
	return (vsl_V4f)_mm_mul_ps(v.vec, u.vec);
}

VSLAPI inline vsl_V4f vsl_v4f_inv(vsl_V4f v) {
	return vsl_v4f_scale(v, -1);
}

VSLAPI inline vsl_V4f vsl_v4f_unit(vsl_V4f v) {
	return vsl_v4f_scale(v, 1.0f/vsl_v4f_len(v));
}

VSLAPI inline float vsl_v4f_dot(vsl_V4f v, vsl_V4f w) {
	vsl_V4f u = (vsl_V4f)_mm_mul_ps(v.vec, w.vec);
	return u.x + u.y + u.z + u.w;
}

VSLAPI inline vsl_V4f vsl_v4f_cross(vsl_V4f v, vsl_V4f w) {
	// V = | 0 | v2 | v1 | v0 | <--- lower
	// W = | 0 | w2 | w1 | w0 | <--- lower

	// V x W = (v1w2 - v2w1,
	//  	    v2w0 - v0w2,
	// 		    v0w1 - v1w0,
	//             0 -    0)

	// V x W = | 0 | v0w1 - v1w0 | v2w0 - v0w2 | v1w2 - v2w1 | <--- lower

	// | 0 | v0 | v2 | v1 |
	__m128 u0 = _mm_shuffle_ps(v.vec, v.vec, VSL_MM_SHUFFLE_MASK(3, 0, 2, 1));
	// | 0 | w1 | w0 | w2 |
	__m128 u1 = _mm_shuffle_ps(w.vec, w.vec, VSL_MM_SHUFFLE_MASK(3, 1, 0, 2));
	// | 0 | v0w2 | v2w1 | v1w0 | <--- crucial step (notice that this mul gives us
	// all factors that come after minus in cross product, just not in the right order).
	__m128 u2 = _mm_mul_ps(u0, w.vec);
	// | 0 | v0w1 | v2w0 | v1w2 |
	__m128 u3 = _mm_mul_ps(u0, u1);
	// | 0 | v1w0 | v0w2 | v2w1 | <--- ordered factors from crucial step
	__m128 u4 = _mm_shuffle_ps(u2, u2, VSL_MM_SHUFFLE_MASK(3, 0, 2, 1));
	// V x W
	return (vsl_V4f)_mm_sub_ps(u3, u4);
}

VSLAPI inline vsl_V4f vsl_v4f_sq(vsl_V4f v) {
	return (vsl_V4f)_mm_mul_ps(v.vec, v.vec);
}

VSLAPI inline float vsl_v4f_sum(vsl_V4f v) {
	return v.x + v.y + v.z + v.w;
}

VSLAPI inline float vsl_v4f_lensq(vsl_V4f v){
	vsl_V4f u = (vsl_V4f)_mm_mul_ps(v.vec, v.vec);
	return u.x + u.y + u.z + u.w;
}

VSLAPI inline float vsl_v4f_len(vsl_V4f v){
	return sqrtf(vsl_v4f_lensq(v));
}

VSLAPI inline void vsl_v4f_add_mut(vsl_V4f *v, vsl_V4f w){
	*v = (vsl_V4f)_mm_add_ps(v->vec, w.vec);
}

VSLAPI inline void vsl_v4f_sub_mut(vsl_V4f *v, vsl_V4f w){
	*v = (vsl_V4f)_mm_sub_ps(v->vec, w.vec);
}

VSLAPI inline void vsl_v4f_mul_mut(vsl_V4f *v, vsl_V4f w){
	*v = (vsl_V4f)_mm_mul_ps(v->vec, w.vec);
}

VSLAPI inline void vsl_v4f_div_mut(vsl_V4f *v, vsl_V4f w){
	*v = (vsl_V4f)_mm_div_ps(v->vec, w.vec);
}

VSLAPI inline void vsl_v4i_add_mut(vsl_V4i *v, vsl_V4i w){
	*v = (vsl_V4i)_mm_add_epi32(v->vec, w.vec);
}

VSLAPI inline void vsl_v4i_sub_mut(vsl_V4i *v, vsl_V4i w){
	*v = (vsl_V4i)_mm_sub_epi32(v->vec, w.vec);
}

VSLAPI inline void vsl_v4f_scale_mut(vsl_V4f *v, float s) {
	vsl_V4f u = {{s, s, s, s}};
	*v = (vsl_V4f)_mm_mul_ps(v->vec, u.vec);
}

VSLAPI inline void vsl_v4f_inv_mut(vsl_V4f *v) {
	vsl_v4f_scale_mut(v, -1);
}

VSLAPI inline void vsl_v4f_unit_mut(vsl_V4f *v) {
	vsl_v4f_scale_mut(v, 1.0f/vsl_v4f_len(*v));
}

VSLAPI inline void vsl_v4f_sq_mut(vsl_V4f *v) {
	*v = (vsl_V4f)_mm_mul_ps(v->vec, v->vec);
}

VSLAPI inline vsl_M4x4f vsl_m4x4f_add(vsl_M4x4f A, vsl_M4x4f B){
	vsl_M4x4f M = {0};
	M.c0 = (vsl_V4f)_mm_add_ps(A.c0.vec, B.c0.vec);
	M.c1 = (vsl_V4f)_mm_add_ps(A.c1.vec, B.c1.vec);
	M.c2 = (vsl_V4f)_mm_add_ps(A.c2.vec, B.c2.vec);
	M.c3 = (vsl_V4f)_mm_add_ps(A.c3.vec, B.c3.vec);
	return M;
}

VSLAPI inline vsl_M4x4f vsl_m4x4f_sub(vsl_M4x4f A, vsl_M4x4f B){
	vsl_M4x4f M = {0};
	M.c0 = (vsl_V4f)_mm_sub_ps(A.c0.vec, B.c0.vec);
	M.c1 = (vsl_V4f)_mm_sub_ps(A.c1.vec, B.c1.vec);
	M.c2 = (vsl_V4f)_mm_sub_ps(A.c2.vec, B.c2.vec);
	M.c3 = (vsl_V4f)_mm_sub_ps(A.c3.vec, B.c3.vec);
	return M;
}

VSLAPI inline vsl_M4x4f vsl_m4x4f_mul(vsl_M4x4f A, vsl_M4x4f B) {
	/*
	  |a0 a1 a2 a3| |x0 y0 z0 w0|
	  |b0 b1 b2 b3| |x1 y1 z1 w1|
	  |c0 c1 c2 c3| |x2 y2 z2 w2|
	  |d0 d1 d2 d3| |x3 y3 z3 w3|

	  Laying out row of the resulting matrix shows patterns, but the problem
	  is that operations in that case would be bad for cache since we store
	  columns in memory, one after the other.

	  Laying out column of the resulting matrix also shows patterns, plus
	  it is adjusted to our way of storing things, so we go with that.

	  This is how it looks for first resulting column.

	  00 = a0x0 + a1x1 + a2x2 + a3x3
	  10 = b0x0 + b1x1 + b2x2 + b3x3
	  20 = c0x0 + c1x1 + c2x2 + c3x3
	  30 = d0x0 + d1x1 + d2x2 + d3x3
	*/
	
	vsl_M4x4f M = {0};
	
	// Result column 0
	__m128 bx = _mm_shuffle_ps(B.c0.vec, B.c0.vec, VSL_MM_SHUFFLE_MASK(0, 0, 0, 0));
	__m128 by = _mm_shuffle_ps(B.c0.vec, B.c0.vec, VSL_MM_SHUFFLE_MASK(1, 1, 1, 1));
	__m128 bz = _mm_shuffle_ps(B.c0.vec, B.c0.vec, VSL_MM_SHUFFLE_MASK(2, 2, 2, 2));
	__m128 bw = _mm_shuffle_ps(B.c0.vec, B.c0.vec, VSL_MM_SHUFFLE_MASK(3, 3, 3, 3));
	__m128 m_0 = _mm_mul_ps(A.c0.vec, bx);
	__m128 m_1 = _mm_mul_ps(A.c1.vec, by);
	__m128 m_2 = _mm_mul_ps(A.c2.vec, bz);
	__m128 m_3 = _mm_mul_ps(A.c3.vec, bw);
	M.c0.vec = _mm_add_ps(_mm_add_ps(m_0, m_1), _mm_add_ps(m_2, m_3));

	// Result column 1
	bx = _mm_shuffle_ps(B.c1.vec, B.c1.vec, VSL_MM_SHUFFLE_MASK(0, 0, 0, 0));
	by = _mm_shuffle_ps(B.c1.vec, B.c1.vec, VSL_MM_SHUFFLE_MASK(1, 1, 1, 1));
	bz = _mm_shuffle_ps(B.c1.vec, B.c1.vec, VSL_MM_SHUFFLE_MASK(2, 2, 2, 2));
	bw = _mm_shuffle_ps(B.c1.vec, B.c1.vec, VSL_MM_SHUFFLE_MASK(3, 3, 3, 3));
	m_0 = _mm_mul_ps(A.c0.vec, bx);
	m_1 = _mm_mul_ps(A.c1.vec, by);
	m_2 = _mm_mul_ps(A.c2.vec, bz);
	m_3 = _mm_mul_ps(A.c3.vec, bw);
	M.c1.vec = _mm_add_ps(_mm_add_ps(m_0, m_1), _mm_add_ps(m_2, m_3));

	// Result column 2
	bx = _mm_shuffle_ps(B.c2.vec, B.c2.vec, VSL_MM_SHUFFLE_MASK(0, 0, 0, 0));
	by = _mm_shuffle_ps(B.c2.vec, B.c2.vec, VSL_MM_SHUFFLE_MASK(1, 1, 1, 1));
	bz = _mm_shuffle_ps(B.c2.vec, B.c2.vec, VSL_MM_SHUFFLE_MASK(2, 2, 2, 2));
	bw = _mm_shuffle_ps(B.c2.vec, B.c2.vec, VSL_MM_SHUFFLE_MASK(3, 3, 3, 3));
	m_0 = _mm_mul_ps(A.c0.vec, bx);
	m_1 = _mm_mul_ps(A.c1.vec, by);
	m_2 = _mm_mul_ps(A.c2.vec, bz);
	m_3 = _mm_mul_ps(A.c3.vec, bw);
	M.c2.vec = _mm_add_ps(_mm_add_ps(m_0, m_1), _mm_add_ps(m_2, m_3));

	// Result column 3
	bx = _mm_shuffle_ps(B.c3.vec, B.c3.vec, VSL_MM_SHUFFLE_MASK(0, 0, 0, 0));
	by = _mm_shuffle_ps(B.c3.vec, B.c3.vec, VSL_MM_SHUFFLE_MASK(1, 1, 1, 1));
	bz = _mm_shuffle_ps(B.c3.vec, B.c3.vec, VSL_MM_SHUFFLE_MASK(2, 2, 2, 2));
	bw = _mm_shuffle_ps(B.c3.vec, B.c3.vec, VSL_MM_SHUFFLE_MASK(3, 3, 3, 3));
	m_0 = _mm_mul_ps(A.c0.vec, bx);
	m_1 = _mm_mul_ps(A.c1.vec, by);
	m_2 = _mm_mul_ps(A.c2.vec, bz);
	m_3 = _mm_mul_ps(A.c3.vec, bw);
	M.c3.vec = _mm_add_ps(_mm_add_ps(m_0, m_1), _mm_add_ps(m_2, m_3));

	return M;
}

VSLAPI inline vsl_M4x4f vsl_m4x4f_scale(vsl_M4x4f A, float s){
	vsl_V4f v = {{s, s, s, s}};
	vsl_M4x4f M = {0};
	M.c0 = (vsl_V4f)_mm_mul_ps(A.c0.vec, v.vec);
	M.c1 = (vsl_V4f)_mm_mul_ps(A.c1.vec, v.vec);
	M.c2 = (vsl_V4f)_mm_mul_ps(A.c2.vec, v.vec);
	M.c3 = (vsl_V4f)_mm_mul_ps(A.c3.vec, v.vec);
	return M;
}

VSLAPI inline vsl_M4x4f vsl_m4x4f_transpose(vsl_M4x4f A) {
	vsl_M4x4f M = {0};
	
	/*
	  Main problem here is that we can't do anything in parallel if we take single
	  column. If we look at first two columns, we know that they need to become rows.
	  So, one thing that we can do is divide matrix into blocks (drawn below).
	  We will store first block info (b f a e) into one vector and second block info
	  (d h c g) info into second vector by using shuffle since it allows us to operate
	  on two vectors at the same time. Order doesn't really matter as long as we
	  know how we coded it and as long as given order can be created with shuffle.
	  For my shuffle, given order is most natural, at least for me.
	  Then, we can use two vectors that we just created and operate twice with shuffle
	  on them to create two resulting columns.
	  We do the same for other blocks.

	  I can't think of something else for now that is also parallel like this.
	
	  Starting point
	  |a b | c d|
	  |e f | g h|
	  -----------
	  |i j | k l|
	  |m n | o p|

	  Intended result
	  |a e i m|
	  |b f j n|
	  |c g k o|
	  |d h l p|
	*/
	
	// b0 = (b f a e)
	__m128 b0 = _mm_shuffle_ps(A.c0.vec, A.c1.vec, VSL_MM_SHUFFLE_MASK(0, 1, 0, 1));
	// b1 = (d h c g)
	__m128 b1 = _mm_shuffle_ps(A.c2.vec, A.c3.vec, VSL_MM_SHUFFLE_MASK(0, 1, 0, 1));
	// b2 = (j n i m)
	__m128 b2 = _mm_shuffle_ps(A.c0.vec, A.c1.vec, VSL_MM_SHUFFLE_MASK(2, 3, 2, 3));
	// b3 = (l p k o)
	__m128 b3 = _mm_shuffle_ps(A.c2.vec, A.c3.vec, VSL_MM_SHUFFLE_MASK(2, 3, 2, 3));
	
	M.c0.vec = _mm_shuffle_ps(b1, b0, VSL_MM_SHUFFLE_MASK(1, 3, 1, 3));
	M.c1.vec = _mm_shuffle_ps(b1, b0, VSL_MM_SHUFFLE_MASK(0, 2, 0, 2));
	M.c2.vec = _mm_shuffle_ps(b3, b2, VSL_MM_SHUFFLE_MASK(1, 3, 1, 3));
	M.c3.vec = _mm_shuffle_ps(b3, b2, VSL_MM_SHUFFLE_MASK(0, 2, 0, 2));

	return M;
}

VSLAPI inline vsl_M4x4f vsl_m4x4f_inv(vsl_M4x4f A) {
	VSL_NOT_IMPLEMENTED("");
}

VSLAPI inline vsl_V4f vsl_m4x4f_map(vsl_V4f v, vsl_M4x4f A) {
	/*
	  |o o o o| |a|                                       |ao bo co do|
	  |o o o o| |b|                                       |ao bo co do|
	  |o o o o| |c|                                       |ao bo co do|
	  |o o o o| |d| ---Expand the vector and the muls---> |ao bo co do|
	*/
	
	__m128 mc0 = _mm_mul_ps(A.c0.vec, _mm_shuffle_ps(v.vec, v.vec, VSL_MM_SHUFFLE_MASK(0, 0, 0, 0)));
	__m128 mc1 = _mm_mul_ps(A.c1.vec, _mm_shuffle_ps(v.vec, v.vec, VSL_MM_SHUFFLE_MASK(1, 1, 1, 1)));
	__m128 mc2 = _mm_mul_ps(A.c2.vec, _mm_shuffle_ps(v.vec, v.vec, VSL_MM_SHUFFLE_MASK(2, 2, 2, 2)));
	__m128 mc3 = _mm_mul_ps(A.c3.vec, _mm_shuffle_ps(v.vec, v.vec, VSL_MM_SHUFFLE_MASK(3, 3, 3, 3)));
	
	/*
	  Horizontal add ---->

	  |ao|   |bo|   |co|   |do|
	  |ao|   |bo|   |co|   |do|
	  |ao|   |bo|   |co|   |do|
	  |ao| + |bo| + |co| + |do|

	  One horizontal sum is the corresponding coordinate of the resulting vector.

	  Not sure if this is the case, but I feel like this kind of addition is slower
	  that summing vectors in pairs and then summing the results, because here, every
	  addition is dependent on previous result, thus stalling processing more.

	  return _mm_add_ps(_mm_add_ps(_mm_add_ps(mc0, mc1), mc2), mc3);
	*/
	
	return (vsl_V4f)_mm_add_ps(_mm_add_ps(mc0, mc1), _mm_add_ps(mc2, mc3));
}

VSLAPI inline float vsl_m4x4f_det(vsl_M4x4f A) {
	VSL_NOT_IMPLEMENTED("");
}

VSLAPI inline float vsl_m4x4f_sum(vsl_M4x4f A) {
	VSL_NOT_IMPLEMENTED("");
	
}

VSLAPI inline void vsl_m4x4f_add_mut(vsl_M4x4f *A, vsl_M4x4f B){
	A->c0 = (vsl_V4f)_mm_add_ps(A->c0.vec, B.c0.vec);
	A->c1 = (vsl_V4f)_mm_add_ps(A->c1.vec, B.c1.vec);
	A->c2 = (vsl_V4f)_mm_add_ps(A->c2.vec, B.c2.vec);
	A->c3 = (vsl_V4f)_mm_add_ps(A->c3.vec, B.c3.vec);
}

VSLAPI inline void vsl_m4x4f_sub_mut(vsl_M4x4f *A, vsl_M4x4f B){
	A->c0 = (vsl_V4f)_mm_sub_ps(A->c0.vec, B.c0.vec);
	A->c1 = (vsl_V4f)_mm_sub_ps(A->c1.vec, B.c1.vec);
	A->c2 = (vsl_V4f)_mm_sub_ps(A->c2.vec, B.c2.vec);
	A->c3 = (vsl_V4f)_mm_sub_ps(A->c3.vec, B.c3.vec);
}

VSLAPI inline void vsl_m4x4f_mul_mut(vsl_M4x4f *A, vsl_M4x4f B) {
	// Since we are modifying matrix A, but also using it in further calculation,
	// we need some temporary storage for original values of A.
	// In order to avoid creating additional matrix on the stack, we use
	// B matrix such that we store column of A (before its usage) in column of
	// B that was already used in calculation and won't be used after it.
	
	// Result column 0
	__m128 bx = _mm_shuffle_ps(B.c0.vec, B.c0.vec, VSL_MM_SHUFFLE_MASK(0, 0, 0, 0));
	__m128 by = _mm_shuffle_ps(B.c0.vec, B.c0.vec, VSL_MM_SHUFFLE_MASK(1, 1, 1, 1));
	__m128 bz = _mm_shuffle_ps(B.c0.vec, B.c0.vec, VSL_MM_SHUFFLE_MASK(2, 2, 2, 2));
	__m128 bw = _mm_shuffle_ps(B.c0.vec, B.c0.vec, VSL_MM_SHUFFLE_MASK(3, 3, 3, 3));
	__m128 m_0 = _mm_mul_ps(A->c0.vec, bx);
	__m128 m_1 = _mm_mul_ps(A->c1.vec, by);
	__m128 m_2 = _mm_mul_ps(A->c2.vec, bz);
	__m128 m_3 = _mm_mul_ps(A->c3.vec, bw);
	B.c0.vec = A->c0.vec;
	A->c0.vec = _mm_add_ps(_mm_add_ps(m_0, m_1), _mm_add_ps(m_2, m_3));

	// Result column 1
	bx = _mm_shuffle_ps(B.c1.vec, B.c1.vec, VSL_MM_SHUFFLE_MASK(0, 0, 0, 0));
	by = _mm_shuffle_ps(B.c1.vec, B.c1.vec, VSL_MM_SHUFFLE_MASK(1, 1, 1, 1));
	bz = _mm_shuffle_ps(B.c1.vec, B.c1.vec, VSL_MM_SHUFFLE_MASK(2, 2, 2, 2));
	bw = _mm_shuffle_ps(B.c1.vec, B.c1.vec, VSL_MM_SHUFFLE_MASK(3, 3, 3, 3));
	m_0 = _mm_mul_ps(B.c0.vec, bx);
	m_1 = _mm_mul_ps(A->c1.vec, by);
	m_2 = _mm_mul_ps(A->c2.vec, bz);
	m_3 = _mm_mul_ps(A->c3.vec, bw);
	B.c1.vec = A->c1.vec;
	A->c1.vec = _mm_add_ps(_mm_add_ps(m_0, m_1), _mm_add_ps(m_2, m_3));

	// Result column 2
	bx = _mm_shuffle_ps(B.c2.vec, B.c2.vec, VSL_MM_SHUFFLE_MASK(0, 0, 0, 0));
	by = _mm_shuffle_ps(B.c2.vec, B.c2.vec, VSL_MM_SHUFFLE_MASK(1, 1, 1, 1));
	bz = _mm_shuffle_ps(B.c2.vec, B.c2.vec, VSL_MM_SHUFFLE_MASK(2, 2, 2, 2));
	bw = _mm_shuffle_ps(B.c2.vec, B.c2.vec, VSL_MM_SHUFFLE_MASK(3, 3, 3, 3));
	m_0 = _mm_mul_ps(B.c0.vec, bx);
	m_1 = _mm_mul_ps(B.c1.vec, by);
	m_2 = _mm_mul_ps(A->c2.vec, bz);
	m_3 = _mm_mul_ps(A->c3.vec, bw);
	B.c2.vec = A->c2.vec;
	A->c2.vec = _mm_add_ps(_mm_add_ps(m_0, m_1), _mm_add_ps(m_2, m_3));

	// Result column 3
	bx = _mm_shuffle_ps(B.c3.vec, B.c3.vec, VSL_MM_SHUFFLE_MASK(0, 0, 0, 0));
	by = _mm_shuffle_ps(B.c3.vec, B.c3.vec, VSL_MM_SHUFFLE_MASK(1, 1, 1, 1));
	bz = _mm_shuffle_ps(B.c3.vec, B.c3.vec, VSL_MM_SHUFFLE_MASK(2, 2, 2, 2));
	bw = _mm_shuffle_ps(B.c3.vec, B.c3.vec, VSL_MM_SHUFFLE_MASK(3, 3, 3, 3));
	m_0 = _mm_mul_ps(B.c0.vec, bx);
	m_1 = _mm_mul_ps(B.c1.vec, by);
	m_2 = _mm_mul_ps(B.c2.vec, bz);
	m_3 = _mm_mul_ps(A->c3.vec, bw);
	A->c3.vec = _mm_add_ps(_mm_add_ps(m_0, m_1), _mm_add_ps(m_2, m_3));
}

VSLAPI inline void vsl_m4x4f_scale_mut(vsl_M4x4f *A, float s){
	vsl_V4f v = {{s, s, s, s}};
	A->c0 = (vsl_V4f)_mm_mul_ps(A->c0.vec, v.vec);
	A->c1 = (vsl_V4f)_mm_mul_ps(A->c1.vec, v.vec);
	A->c2 = (vsl_V4f)_mm_mul_ps(A->c2.vec, v.vec);
	A->c3 = (vsl_V4f)_mm_mul_ps(A->c3.vec, v.vec);
}

VSLAPI inline void vsl_m4x4f_transpose_mut(vsl_M4x4f *A) {
	VSL_NOT_IMPLEMENTED("");
}

VSLAPI inline void vsl_m4x4f_inv_mut(vsl_M4x4f *A) {
	VSL_NOT_IMPLEMENTED("");
}

VSLAPI inline void vsl_m4x4f_map_mut(vsl_V4f *v, vsl_M4x4f A) {
	__m128 mc0 = _mm_mul_ps(A.c0.vec, _mm_shuffle_ps(v->vec, v->vec, VSL_MM_SHUFFLE_MASK(0, 0, 0, 0)));
	__m128 mc1 = _mm_mul_ps(A.c1.vec, _mm_shuffle_ps(v->vec, v->vec, VSL_MM_SHUFFLE_MASK(1, 1, 1, 1)));
	__m128 mc2 = _mm_mul_ps(A.c2.vec, _mm_shuffle_ps(v->vec, v->vec, VSL_MM_SHUFFLE_MASK(2, 2, 2, 2)));
	__m128 mc3 = _mm_mul_ps(A.c3.vec, _mm_shuffle_ps(v->vec, v->vec, VSL_MM_SHUFFLE_MASK(3, 3, 3, 3)));

	*v = (vsl_V4f)_mm_add_ps(_mm_add_ps(mc0, mc1), _mm_add_ps(mc2, mc3));
}



VSLAPI void vsl_v4f_print(vsl_V4f v) {
	printf("[%f, %f, %f, %f]\n", v.v0, v.v1, v.v2, v.v3);
}

VSLAPI void vsl_v4i_print(vsl_V4i v) {
	printf("[%d, %d, %d, %d]\n", v.v0, v.v1, v.v2, v.v3);
}

VSLAPI void vsl_m4x4f_print(vsl_M4x4f A) {
	printf("----------------------------------------\n");
	vsl_v4f_print(A.c0);
	vsl_v4f_print(A.c1);
	vsl_v4f_print(A.c2);
	vsl_v4f_print(A.c3);
	printf("----------------------------------------\n");
}

#endif // VSL_IMPLEMENTATION

// TODO: Test inline functions vs corresponding macros since most functions are
// extremely simple.
// TODO: Test vector instructions usage for cases where they are followed by
// ordinary instructions.
// TODO: Test how explicit loading of wide register impacts performance in
// functions like vector scale (or matrix scale).
// TODO: Test if compiler does one wide register load in cases where something
// like v->vec is used multiple times.
// TODO: Test if compiler does preload if it sees definition of __m128 before usage.
// I feel like this can be detected with static analysis.
// TODO: Test impact of reusage of __m128 vars.
