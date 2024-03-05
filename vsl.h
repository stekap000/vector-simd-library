#ifndef VSL_H
#define VSL_H

// TODO: Write checks for availability of intrinsics.
// For now, we just assume this is valid.
#include <immintrin.h>
#include <stdint.h>

#define VSLAPI static

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



VSLAPI inline void vsl_v4f_print(vsl_V4f v);

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



// TODO: Remove when testing is not needed anymore.
#include <stdio.h>
VSLAPI inline void vsl_v4f_print(vsl_V4f v) {
	printf("[%f, %f, %f, %f]\n", v.v0, v.v1, v.v2, v.v3);
}

#endif // VSL_IMPLEMENTATION
