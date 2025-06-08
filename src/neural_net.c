//
// Created by joshua on 08.06.25.
//
#include <stdlib.h>

#include "neural_net.h"

#include <math.h>

#include "calculations.h"

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

int feedforward(NeuralNet* neural_net, double input[LAYER_1])
{
    for (int i = 0; i < LAYER_1; i++)
    {
        neural_net->layer1->neuronArray[i]->activation = input[i];
    }
    propagate(neural_net->layer1, neural_net->connection1_2, neural_net->layer2);
    propagate(neural_net->layer2, neural_net->connection2_3, neural_net->layer3);
    propagate(neural_net->layer3, neural_net->connection3_4, neural_net->layer4);

    int index = 0;
    int max = 0;
    for (int i = 0; i < LAYER_4; i++)
    {
        if (max < neural_net->layer4->neuronArray[i]->activation)
        {
            max = neural_net->layer4->neuronArray[i]->activation;
            index = i;
        }
    }
    return index;
}


void propagate(NeuronLayer* lhs, ConnectionLayer* conn, NeuronLayer* rhs)
{
    // MATRIX (weights) * activations + bias -> sigmoid (holy fucking shit is this exhausting)
    double* activation_vector = malloc(sizeof(double) * lhs->size);
    for (int i = 0; i < lhs->size; i++)
    {
        activation_vector[i] = lhs->neuronArray[i]->activation;
    }

    double* weight_matrix = malloc(sizeof(double) * conn->size);
    for (int i = 0; i < conn->size; i++)
    {
        weight_matrix[i] = conn->connectionArray[i]->weight;
    }
    double* result_vector = malloc(sizeof(double) * rhs->size);

    for (int i = 0; i < rhs->size; i++)
    {
        double sum = 0;
        for (int j = 0; j < lhs->size; j++)
        {
            sum += activation_vector[j] * weight_matrix[lhs->size*i + j];
        }
        result_vector[i] = sum + rhs->neuronArray[i]->bias;
    }
    for (int i = 0; i < rhs->size; i++)
    {
        rhs->neuronArray[i]->activation = sigmoid(result_vector[i]);
    }
    free(result_vector);
    free(weight_matrix);
    free(activation_vector);
}

double cost_function(NeuralNet* neural_net, int expected_value)
{
    double sum = 0;
    for (int i = 0; i < LAYER_4; i++)
    {
        if (i == expected_value)
        {
            sum += (neural_net->layer4->neuronArray[i]->activation - 1)*(neural_net->layer4->neuronArray[i]->activation - 1);
        }
        else sum += neural_net->layer4->neuronArray[i]->activation*neural_net->layer4->neuronArray[i]->activation;
    }
    return sum;
}

void back_propagate(NeuralNet* neural_net, int expected_value)
{

}
