//
// Created by joshua on 08.06.25.
//

#ifndef APP_H
#define APP_H

#include <SDL2/SDL.h>

#define GRID_SIZE 28
#define CELL_SIZE 20

#define WINDOW_SIZE (CELL_SIZE * GRID_SIZE)



void app();
void draw_grid(SDL_Renderer* renderer);
void apply_brush(int centerX, int centerY);



#endif //APP_H