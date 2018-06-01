
//#ifdef cl_khr_fp64
//    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#elif defined(cl_amd_fp64)
//    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
//#else
//    #warning "Double precision floating point not supported by OpenCL implementation."
//#endif


// NE PAS MODIFIER
static int4 color_to_int4 (unsigned c)
{
  uchar4 ci = *(uchar4 *) &c;
  return convert_int4 (ci);
}

// NE PAS MODIFIER
static unsigned int4_to_color (int4 i)
{
  return (unsigned) convert_uchar4 (i);
}

static unsigned color_mean (unsigned c1, unsigned c2)
{
  return int4_to_color ((color_to_int4 (c1) + color_to_int4 (c2)) / (int4)2);
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// scrollup
////////////////////////////////////////////////////////////////////////////////

__kernel void scrollup (__global unsigned *in, __global unsigned *out)
{
  int y = get_global_id (1);
  int x = get_global_id (0);
  unsigned couleur;

  couleur = in [y * DIM + x];

  y = (y ? y - 1 : get_global_size (1) - 1);

  out [y * DIM + x] = couleur;
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// vie
////////////////////////////////////////////////////////////////////////////////


__kernel void vie (__global unsigned *in, __global unsigned *out)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  if(y > 0 && y < DIM-1 && x > 0 && x < DIM-1){
    int n = 0;

    for (int i = y - 1; i <= y + 1; i++)
    for (int j = x - 1; j <= x + 1; j++)
    n += (in[i*DIM+j] == 0xFFFF00FF);

    if (in[y*DIM+x] == 0xFFFF00FF) {
      if (n == 3 || n == 4)
      n = 0xFFFF00FF;
      else
      n = 0;
    } else {
      if (n == 3)
      n = 0xFFFF00FF;
      else
      n = 0;
    }

    out[y*DIM+x] = n;
  }
}



// NE PAS MODIFIER
static float4 color_scatter (unsigned c)
{
  uchar4 ci;

  ci.s0123 = (*((uchar4 *) &c)).s3210;
  return convert_float4 (ci) / (float4) 255;
}

// NE PAS MODIFIER: ce noyau est appelé lorsqu'une mise à jour de la
// texture de l'image affichée est requise
__kernel void update_texture (__global unsigned *cur, __write_only image2d_t tex)
{
  int y = get_global_id (1);
  int x = get_global_id (0);
  int2 pos = (int2)(x, y);
  unsigned c = cur [y * DIM + x];

  write_imagef (tex, pos, color_scatter (c));
}
