#ifndef VSL_H
#define VSL_H

// TODO: Write checks for availability of intrinsics.
// For now, we just assume this is valid.
#include <immintrin.h>
#include <stdint.h>
#include <assert.h>

#define VSLAPI static

#define VSL_NOT_IMPLEMENTED(msg) assert(!(msg))

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
VSLAPI inline float vsl_v4f_dot(vsl_V4f v, vsl_V4f w);
VSLAPI inline vsl_V4f vsl_v4f_cross(vsl_V4f v, vsl_V4f w);
VSLAPI inline vsl_V4f vsl_v4f_sq(vsl_V4f v);

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

VSLAPI inline float vsl_v4f_dot(vsl_V4f v, vsl_V4f w) {
	VSL_NOT_IMPLEMENTED("Dot product not yet implemented");
	return 0;
}

VSLAPI inline vsl_V4f vsl_v4f_cross(vsl_V4f v, vsl_V4f w) {
	// V = (v0, v1, v2, 0)
	// W = (w0, w1, w2, 0)
	// 	
	// V x W = (v1w2 - v2w1,
	//  	    v2w0 - v0w2,
	// 		    v0w1 - v1w0
	//          0)
	
	// From intel intrinsics doc.
	// Keeping this here for now because I always forget shuffle intrinsic mask positions.
	//DEFINE SELECT4(src, control) {
	//	CASE(control[1:0]) OF
	//	0:	tmp[31:0] := src[31:0]
	//	1:	tmp[31:0] := src[63:32]
	//	2:	tmp[31:0] := src[95:64]
	//	3:	tmp[31:0] := src[127:96]
	//	ESAC
	//	RETURN tmp[31:0]
	//}
	//dst[31:0] := SELECT4(a[127:0], imm8[1:0])
	//dst[63:32] := SELECT4(a[127:0], imm8[3:2])
	//dst[95:64] := SELECT4(b[127:0], imm8[5:4])
	//dst[127:96] := SELECT4(b[127:0], imm8[7:6])
	
	__m128 tmp0 = _mm_shuffle_ps(v.vec, v.vec, VSL_MM_SHUFFLE_MASK(3, 0, 2, 1));
	__m128 tmp1 = _mm_shuffle_ps(w.vec, w.vec, VSL_MM_SHUFFLE_MASK(3, 1, 0, 2));
	__m128 tmp2 = _mm_shuffle_ps(v.vec, v.vec, VSL_MM_SHUFFLE_MASK(3, 1, 0, 2));
	__m128 tmp3 = _mm_shuffle_ps(w.vec, w.vec, VSL_MM_SHUFFLE_MASK(3, 0, 2, 1));
	
	return (vsl_V4f)_mm_sub_ps(_mm_mul_ps(tmp0, tmp1), _mm_mul_ps(tmp2, tmp3));
}

VSLAPI inline vsl_V4f vsl_v4f_sq(vsl_V4f v) {
	return vsl_v4f_mul(v, v);
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
