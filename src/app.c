//
// Created by joshua on 08.06.25.
//
#include "app.h"
#include "neural_net.h"
#include <SDL2/SDL.h>
#include <stdbool.h>


double input[GRID_SIZE][GRID_SIZE] = {0};

void app()
{
    NeuralNet* neural_net = instantiate_neural_net();
    load_wb(neural_net);

    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *window = SDL_CreateWindow(
        "28x28 number drawer:", SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED, WINDOW_SIZE,
        WINDOW_SIZE, SDL_WINDOW_SHOWN);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    bool running = true;
    SDL_Event event;
    while (running)
    {
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
            {
                running = false;
            }
            else if (event.type == SDL_MOUSEBUTTONDOWN || event.type == SDL_MOUSEMOTION)
            {
                if (event.type == SDL_MOUSEBUTTONDOWN ||
                    (event.type == SDL_MOUSEMOTION && (
                         event.motion.state & SDL_BUTTON_LMASK || event.motion.state & SDL_BUTTON_RMASK)))
                {
                    int x = event.motion.x / CELL_SIZE;
                    int y = event.motion.y / CELL_SIZE;
                    if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE)
                    {
                        if (event.button.button == SDL_BUTTON_LEFT)
                        {
                            apply_brush(x, y);
                        } else if (event.button.button == SDL_BUTTON_RIGHT)
                        {
                            input[y][x] = 0.0;
                        }
                    }
                }
            }
            else if (event.type == SDL_KEYDOWN)
            {
                if (event.key.keysym.sym == SDLK_RETURN)
                {
                    double flat_input[784];
                    for (int y = 0; y < GRID_SIZE; y++)
                    {
                        for (int x = 0; x < GRID_SIZE; x++)
                        {
                            flat_input[y * GRID_SIZE + x] = input[y][x];
                        }
                    }

                    //Call feedforward function here:
                    int result = feedforward(neural_net, flat_input);
                    if (result == 11) printf("unsure\n");
                    else printf("number: %d\n", result);
                }

                if (event.key.keysym.sym == SDLK_c)
                {
                    for (int y = 0; y < GRID_SIZE; y++)
                    {
                        for (int x = 0; x < GRID_SIZE; x++)
                        {
                            input[y][x] = 0.0;
                        }
                    }
                }
            }
        }
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        SDL_RenderClear(renderer);

        draw_grid(renderer);

        SDL_RenderPresent(renderer);
    }
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    save_wb(neural_net);
    close_neural_net(neural_net);
}

void draw_grid(SDL_Renderer *renderer)
{
    for (int y = 0; y < GRID_SIZE; y++)
    {
        for (int x = 0; x < GRID_SIZE; x++)
        {
            SDL_Rect cell = {x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE};

            int value = (int) ((1.0 - input[y][x]) * 255);
            if (value < 0) value = 0;
            if (value > 255) value = 255;

            SDL_SetRenderDrawColor(renderer, value, value, value, 255);
            SDL_RenderFillRect(renderer, &cell);

            SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
            SDL_RenderDrawRect(renderer, &cell);
        }
    }
}

void apply_brush(int centerX, int centerY)
{
    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            int x = centerX + dx;
            int y = centerY + dy;

            if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE)
            {
                double distance = abs(dx) + abs(dy);
                double value = 0.0;

                if (distance == 0)
                    value = 1.0;
                else if (distance == 1)
                    value = 0.3;
                else if (distance == 2)
                    value = 0.1;

                if (input[y][x] < value)
                    input[y][x] = value;
            }
        }
    }
}

