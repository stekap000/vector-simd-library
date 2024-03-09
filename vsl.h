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


VSLAPI inline vsl_V4f vsl_v4f_add(vsl_V4f v, vsl_V4f w);
VSLAPI inline vsl_V4f vsl_v4f_sub(vsl_V4f v, vsl_V4f w);
VSLAPI inline vsl_V4f vsl_v4f_mul(vsl_V4f v, vsl_V4f w);
VSLAPI inline vsl_V4f vsl_v4f_div(vsl_V4f v, vsl_V4f w);
VSLAPI inline vsl_V4i vsl_v4i_add(vsl_V4i v, vsl_V4i w);
VSLAPI inline vsl_V4i vsl_v4i_sub(vsl_V4i v, vsl_V4i w);

VSLAPI inline vsl_V4f vsl_v4f_scale(vsl_V4f v, float s);
VSLAPI inline vsl_V4f vsl_v4f_unit(vsl_V4f v);
VSLAPI inline float vsl_v4f_dot(vsl_V4f v, vsl_V4f w);
VSLAPI inline vsl_V4f vsl_v4f_cross(vsl_V4f v, vsl_V4f w);
VSLAPI inline vsl_V4f vsl_v4f_sq(vsl_V4f v);
VSLAPI inline float vsl_v4f_sum(vsl_V4f v);
VSLAPI inline float vsl_v4f_lensq(vsl_V4f v);
VSLAPI inline float vsl_v4f_len(vsl_V4f v);

VSLAPI inline void vsl_v4f_add_mut(vsl_V4f *v, vsl_V4f w);
VSLAPI inline void vsl_v4f_sub_mut(vsl_V4f *v, vsl_V4f w);
VSLAPI inline void vsl_v4f_mul_mut(vsl_V4f *v, vsl_V4f w);
VSLAPI inline void vsl_v4f_div_mut(vsl_V4f *v, vsl_V4f w);
VSLAPI inline void vsl_v4i_add_mut(vsl_V4i *v, vsl_V4i w);
VSLAPI inline void vsl_v4i_sub_mut(vsl_V4i *v, vsl_V4i w);

VSLAPI inline void vsl_v4f_scale_mut(vsl_V4f *v, float s);
VSLAPI inline void vsl_v4f_unit_mut(vsl_V4f *v);
VSLAPI inline void vsl_v4f_sq_mut(vsl_V4f *v);

VSLAPI inline void vsl_v4f_print(vsl_V4f v);
VSLAPI inline void vsl_v4i_print(vsl_V4i v);

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

VSLAPI inline void vsl_v4f_unit_mut(vsl_V4f *v) {
	vsl_v4f_scale_mut(v, 1.0f/vsl_v4f_len(*v));
}

VSLAPI inline void vsl_v4f_sq_mut(vsl_V4f *v) {
	*v = (vsl_V4f)_mm_mul_ps(v->vec, v->vec);
}

VSLAPI inline void vsl_v4f_print(vsl_V4f v) {
	printf("[%f, %f, %f, %f]\n", v.v0, v.v1, v.v2, v.v3);
}

VSLAPI inline void vsl_v4i_print(vsl_V4i v) {
	printf("[%d, %d, %d, %d]\n", v.v0, v.v1, v.v2, v.v3);
}

#endif // VSL_IMPLEMENTATION

// TODO: Test inline functions vs corresponding macros since most functions are
// extremely simple.
// TODO: Test vector instructions usage for cases where they are followed by
// ordinary instructions.
// TODO: Test how explicit loading of wide register impacts performance in
// functions like vector scale.
