
#include "global.h"
#include "compute.h"
#include "graphics.h"
#include "debug.h"
#include "ocl.h"
#include "scheduler.h"

#include <stdbool.h>


///////////////////////////// Version séquentielle simple (seq)

static void compute_new_state (int y, int x)
{
  unsigned n = 0;

  if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1) {

    for (int i = y - 1; i <= y + 1; i++)
    for (int j = x - 1; j <= x + 1; j++)
    n += (cur_img (i, j) != 0);

    if (cur_img (y, x) != 0) {
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

    next_img (y, x) = n;
  }
}


// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned vie_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it ++) {

    for (int i = 0; i < DIM; i++)
    for (int j = 0; j < DIM; j++)
    compute_new_state (i, j);

    swap_images ();
  }

  return 0;
}


///////////////////////////// Version séquentielle tuilée (tile)

unsigned vie_compute_tile (unsigned nb_iter)
{
  TILEX = DIM/GRAIN;
  TILEY = DIM/GRAIN;
  for (unsigned it = 1; it <= nb_iter; it ++) {
    for (int x = 0; x < GRAIN; x++) {
      for (int y = 0; y < GRAIN; y++) {
        for (int i = TILEX*x; i < TILEX*(x+1); i++) {
          for (int j = TILEY*y; j < TILEY*(y+1); j++) {
            compute_new_state (i, j);
          }
        }
      }
    }

    swap_images ();
  }

  return 0;
}


///////////////////////////// Version séquentielle optimisée (opti)

unsigned vie_compute_opti (unsigned nb_iter)
{
  TILEX = DIM/GRAIN;
  TILEY = DIM/GRAIN;
  for (unsigned it = 1; it <= nb_iter; it ++) {
    for (int x = 0; x < GRAIN; x++) {
      for (int y = 0; y < GRAIN; y++) {
        for (int i = TILEX*x; i < TILEX*(x+1); i++) {
          for (int j = TILEY*y; j < TILEY*(y+1); j++) {
            compute_new_state (i, j);
          }
        }
      }
    }

    swap_images ();
  }

  return 0;
}


///////////////////////////// Configuration initiale

void draw_stable (void);
void draw_guns (void);
void draw_random (void);

void vie_draw (char *param)
{
  char func_name [1024];
  void (*f)(void) = NULL;

  sprintf (func_name, "draw_%s", param);
  f = dlsym (DLSYM_FLAG, func_name);

  if (f == NULL) {
    printf ("Cannot resolve draw function: %s\n", func_name);
    f = draw_guns;
  }

  f ();
}

static unsigned couleur = 0xFFFF00FF; // Yellow

static void gun (int x, int y, int version)
{
  bool glider_gun [11][38] =
  {
    { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
    { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0 },
    { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0 },
    { 0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0 },
    { 0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0 },
    { 0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
    { 0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0 },
    { 0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0 },
    { 0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
    { 0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
    { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
  };

  if (version == 0)
  for (int i=0; i < 11; i++)
  for(int j=0; j < 38; j++)
  if (glider_gun [i][j])
  cur_img (i+x, j+y) = couleur;

  if (version == 1)
  for (int i=0; i < 11; i++)
  for(int j=0; j < 38; j++)
  if (glider_gun [i][j])
  cur_img (x-i, j+y) = couleur;

  if (version == 2)
  for (int i=0; i < 11; i++)
  for(int j=0; j < 38; j++)
  if (glider_gun [i][j])
  cur_img (x-i, y-j) = couleur;

  if (version == 3)
  for (int i=0; i < 11; i++)
  for(int j=0; j < 38; j++)
  if (glider_gun [i][j])
  cur_img (i+x, y-j) = couleur;

}

void draw_stable (void)
{
  for (int i=1; i < DIM-2; i+=4)
  for(int j=1; j < DIM-2; j+=4)
  cur_img (i, j) = cur_img (i, (j+1)) =cur_img ((i+1), j) =cur_img ((i+1), (j+1)) = couleur;
}

void draw_guns (void)
{
  memset(&cur_img (0,0), 0, DIM*DIM* sizeof(cur_img (0,0)));

  gun (0, 0, 0);
  gun (0,  DIM-1 , 3);
  gun (DIM - 1 , DIM - 1, 2);
  gun (DIM - 1 , 0, 1);
}

void draw_random (void)
{
  for (int i=1; i < DIM-1; i++)
  for(int j=1; j < DIM-1; j++)
  cur_img (i, j) = (random() & 01) ? couleur : 0;
}

// Une tête de clown apparaît à l'itération 110
void draw_clown (void)
{
  int i = DIM/2, j = i;

  cur_img (i, j-1) = couleur;
  cur_img (i, j) = couleur;
  cur_img (i, j+1) = couleur;

  cur_img (i+1, j-1) = couleur;
  cur_img (i+1, j+1) = couleur;

  cur_img (i+2, j-1) = couleur;
  cur_img (i+2, j+1) = couleur;
}
