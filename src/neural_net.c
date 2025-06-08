//
// Created by joshua on 08.06.25.
//
#include <stdlib.h>

#include "../include/neural_net.h"

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
    ConnectionLayer *layer = malloc(sizeof(ConnectionLayer));
    layer->size = layer1->size * layer2->size;
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