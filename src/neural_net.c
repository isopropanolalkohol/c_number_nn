//
// Created by joshua on 08.06.25.
//
#include <stdlib.h>

#include "../include/neural_net.h"

#include <stdio.h>
#include <string.h>

NeuralNet* instantiate_neural_net()
{
    NeuralNet *net = malloc(sizeof(NeuralNet));
    net->layer1 = instantiate_neuron_layer(LAYER_1);
    net->layer2 = instantiate_neuron_layer(LAYER_2);
    net->layer3 = instantiate_neuron_layer(LAYER_3);
    net->layer4 = instantiate_neuron_layer(LAYER_4);

    net->connection1_2 = instantiate_connection_layer(net->layer1, net->layer2);
    net->connection2_3 = instantiate_connection_layer(net->layer2, net->layer3);
    net->connection3_4 = instantiate_connection_layer(net->layer3, net->layer4);

    return net;
}

void close_neural_net(NeuralNet *neural_net)
{
    //free all the other shit
    free_connection_layer(neural_net->connection1_2);
    free_connection_layer(neural_net->connection2_3);
    free_connection_layer(neural_net->connection3_4);

    free_neuron_layer(neural_net->layer1);
    free_neuron_layer(neural_net->layer2);
    free_neuron_layer(neural_net->layer3);
    free_neuron_layer(neural_net->layer4);

    //free the net
    free(neural_net);
}

NeuronLayer* instantiate_neuron_layer(const int neuronCount)
{
    NeuronLayer *layer = malloc(sizeof(NeuronLayer));
    Neuron** array = malloc(sizeof(Neuron*) * neuronCount);
    layer->size = neuronCount;
    for (int i = 0; i < neuronCount; i++)
    {
        Neuron* neuron = malloc(sizeof(Neuron));
        neuron->activation = 0;
        neuron->bias = 0;
        array[i] = neuron;
    }
    layer->neuronArray = array;
    return layer;
}

void free_neuron_layer(NeuronLayer *layer)
{
    for (int i = 0; i < layer->size; i++)
    {
        free(layer->neuronArray[i]);
    }
    free(layer->neuronArray);
    free(layer);
}

ConnectionLayer* instantiate_connection_layer(NeuronLayer* layer1, NeuronLayer* layer2)
{
    // saved in the following: for each neuron in layer 2: all neuron in layer 1
    ConnectionLayer *layer = malloc(sizeof(ConnectionLayer));
    layer->size = layer1->size * layer2->size;
    layer->input_size = layer1->size;
    Connection** array = malloc(sizeof(Connection*) * layer->size);
    for (int i = 0; i < layer2->size; i++)
    {
        for (int j = 0; j < layer1->size; j++)
        {
            Connection* conn = malloc(sizeof(Connection));
            conn->source = layer1->neuronArray[j];
            conn->dest = layer2->neuronArray[i];
            conn->weight = 0;
            array[layer1->size*i + j] = conn;
        }
    }
    layer->connectionArray = array;
    return layer;
}

void free_connection_layer(ConnectionLayer* layer)
{
    for (int i = 0; i < layer->size; i++)
    {
        free(layer->connectionArray[i]);
    }
    free(layer->connectionArray);
    free(layer);
}


void load_wb(NeuralNet* neural_net)
{
    read_weights(FILEPATH_CONNLAYER1, neural_net->connection1_2);
    read_weights(FILEPATH_CONNLAYER2, neural_net->connection2_3);
    read_weights(FILEPATH_CONNLAYER3, neural_net->connection3_4);

    read_biases(FILEPATH_NEURLAYER2, neural_net->layer2);
    read_biases(FILEPATH_NEURLAYER3, neural_net->layer3);
    read_biases(FILEPATH_NEURLAYER4, neural_net->layer4);
}
void save_wb(NeuralNet* neural_net)
{
    write_weights(FILEPATH_CONNLAYER1, neural_net->connection1_2);
    write_weights(FILEPATH_CONNLAYER2, neural_net->connection2_3);
    write_weights(FILEPATH_CONNLAYER3, neural_net->connection3_4);

    write_biases(FILEPATH_NEURLAYER2, neural_net->layer2);
    write_biases(FILEPATH_NEURLAYER3, neural_net->layer3);
    write_biases(FILEPATH_NEURLAYER4, neural_net->layer4);
}

void write_weights(const char* filepath, ConnectionLayer* layer)
{
    FILE *file = fopen(filepath, "w");
    if (file == NULL)
    {
        perror("ERROR: Can't open weight file.");
        return;
    }
    for (int i = 0; i < layer->size/layer->input_size; i++)
    {
        for (int j = 0; j < layer->input_size; j++)
        {
            fprintf(file, "%+07.3f ", layer->connectionArray[layer->input_size*i + j]->weight);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}
void read_weights(const char* filepath, ConnectionLayer* layer)
{
    char buffer[8192];
    FILE *file = fopen(filepath, "r");
    if (file == NULL)
    {
        perror("ERROR: Can't open weight file.");
        return;
    }
    int j = 0;
    while (fgets(buffer, 8192, file))
    {
        char* token = strtok(buffer, " ");
        int i = 0;

        while (token != NULL && i < layer->input_size) {
            layer->connectionArray[layer->input_size*j + i]->weight = strtod(token, NULL);
            token = strtok(NULL, " ");
            i++;
        }
        j++;
    }
    fclose(file);
}

void write_biases(const char* filepath, NeuronLayer* layer)
{
    FILE *file = fopen(filepath, "w");
    if (file == NULL)
    {
        perror("ERROR: Can't open bias file.");
        return;
    }
    for (int i = 0; i < layer->size; i++)
    {
        fprintf(file, "%+07.3f\n", layer->neuronArray[i]->bias);
    }
    fclose(file);
}
void read_biases(const char* filepath, const NeuronLayer* layer)
{
    FILE *file = fopen(filepath, "r");
    if (file == NULL)
    {
        perror("ERROR: Can't open bias file.");
        return;
    }
    char buffer[20];
    int i = 0;
    while (fgets(buffer, 20, file))
    {
        layer->neuronArray[i]->bias = strtod(buffer, NULL);
        i++;
    }
    fclose(file);
}