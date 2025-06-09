//
// Created by joshua on 08.06.25.
//
#include <stdlib.h>

#include "neural_net.h"

#include <math.h>

#include "calculations.h"

#include <stdio.h>
#include <string.h>

#include "app.h"

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
    //printf("z1: %03f, %03f, %03f\n", neural_net->layer1->neuronArray[0]->activation, neural_net->layer1->neuronArray[1]->activation, neural_net->layer1->neuronArray[2]->activation);
    //printf("a1: %03f, %03f, %03f\n", neural_net->layer2->neuronArray[0]->activation, neural_net->layer2->neuronArray[1]->activation, neural_net->layer2->neuronArray[2]->activation);
    propagate(neural_net->layer2, neural_net->connection2_3, neural_net->layer3);
    //printf("z2: %03f, %03f, %03f\n", neural_net->layer2->neuronArray[0]->activation, neural_net->layer2->neuronArray[1]->activation, neural_net->layer2->neuronArray[2]->activation);
    //printf("a2: %03f, %03f, %03f\n", neural_net->layer3->neuronArray[0]->activation, neural_net->layer3->neuronArray[1]->activation, neural_net->layer3->neuronArray[2]->activation);
    propagate(neural_net->layer3, neural_net->connection3_4, neural_net->layer4);
    //printf("z3: %03f, %03f, %03f\n", neural_net->layer3->neuronArray[0]->activation, neural_net->layer3->neuronArray[1]->activation, neural_net->layer3->neuronArray[2]->activation);
    //printf("a3: %03f, %03f, %03f\n", neural_net->layer4->neuronArray[0]->activation, neural_net->layer4->neuronArray[1]->activation, neural_net->layer4->neuronArray[2]->activation);
    int index = 0;
    double max = 0;
    for (int i = 0; i < LAYER_4; i++)
    {
        if (max < neural_net->layer4->neuronArray[i]->activation)
        {
            max = neural_net->layer4->neuronArray[i]->activation;
            index = i;
        }
    }
    if (max < 0.5) return 11;
    return index;
}


void propagate(NeuronLayer* lhs, ConnectionLayer* conn, NeuronLayer* rhs)
{
    double* result_vector = z_l(lhs, conn, rhs);
    for (int i = 0; i < rhs->size; i++)
    {

        rhs->neuronArray[i]->activation = sigmoid(result_vector[i]);
    }
    free(result_vector);
}

double cost_function(const NeuralNet* neural_net, const int expected_value)
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

void back_propagate(const NeuralNet* neural_net, const int expected_value)
{
    gradCf* gradient = derivative(neural_net, expected_value);
    for (int i = 0; i < LAYER_4; i++)
    {
        gradient->gradBiases_l3[i] *= ETA;
    }
    for (int i = 0; i < LAYER_4; i++)
    {
        neural_net->layer4->neuronArray[i]->bias -= gradient->gradBiases_l3[i];
    }
    for (int i = 0; i < LAYER_3; i++)
    {
        gradient->gradBiases_l2[i] *= ETA;
    }
    for (int i = 0; i < LAYER_3; i++)
    {
        neural_net->layer3->neuronArray[i]->bias -= gradient->gradBiases_l2[i];
    }
    for (int i = 0; i < LAYER_2; i++)
    {
        gradient->gradBiases_l1[i] *= ETA;
    }
    for (int i = 0; i < LAYER_2; i++)
    {
        neural_net->layer2->neuronArray[i]->bias -= gradient->gradBiases_l1[i];
    }
    for (int i = 0; i < neural_net->connection1_2->size; i++)
    {
        gradient->gradWeights_l1[i] *= ETA;
    }
    for (int i = 0; i < neural_net->connection1_2->size; i++)
    {
        neural_net->connection1_2->connectionArray[i]->weight -= gradient->gradWeights_l1[i];
    }
    for (int i = 0; i < neural_net->connection2_3->size; i++)
    {
        gradient->gradWeights_l2[i] *= ETA;
    }
    for (int i = 0; i < neural_net->connection2_3->size; i++)
    {
        neural_net->connection2_3->connectionArray[i]->weight -= gradient->gradWeights_l2[i];
    }
    for (int i = 0; i < neural_net->connection3_4->size; i++)
    {
        gradient->gradWeights_l3[i] *= ETA;
    }
    for (int i = 0; i < neural_net->connection3_4->size; i++)
    {
        neural_net->connection3_4->connectionArray[i]->weight -= gradient->gradWeights_l3[i];
    }
    free(gradient->gradBiases_l3);
    free(gradient->gradBiases_l2);
    free(gradient->gradBiases_l1);
    free(gradient->gradWeights_l3);
    free(gradient->gradWeights_l2);
    free(gradient->gradWeights_l1);
    free(gradient);

    double mse = 0.0;
    double max_activation = 0.0;
    int max_index = 0;

    for (int i = 0; i < LAYER_4; i++) {
        double a = neural_net->layer4->neuronArray[i]->activation;
        double y = (i == expected_value) ? 1.0 : 0.0;

        double diff = a - y;
        mse += diff * diff;

        if (a > max_activation) {
            max_activation = a;
            max_index = i;
        }
    }

    mse /= LAYER_4;

    printf("Prediction: %d | Expected: %d | Max a: %.4f | MSE: %.6f\n",
           max_index, expected_value, max_activation, mse);

}


void train(NeuralNet* neural_net, const char* filepath_image, const char* filepath_label)
{
    FILE *images = fopen(filepath_image, "r");
    if (images == NULL)
    {
        perror("ERROR: Can't open image file.");
        return;
    }
    FILE *label = fopen(filepath_label, "r");
    if (label == NULL)
    {
        perror("ERROR: Can't open label file.");
        return;
    }
    u_int8_t expected_number;
    double image[GRID_SIZE][GRID_SIZE] = {0};
    uint32_t magic, num_images, rows, cols;

    fread(&magic, 4, 1, images);
    fread(&num_images, 4, 1, images);
    fread(&rows, 4, 1, images);
    fread(&cols, 4, 1, images);
    magic = reverse_uint32(magic);
    num_images = reverse_uint32(num_images);
    rows = reverse_uint32(rows);
    cols = reverse_uint32(cols);

    uint32_t magic2, num_labels;
    fread(&magic2, 4, 1, label);
    fread(&num_labels, 4, 1, label);

    magic2 = reverse_uint32(magic2);
    num_labels = reverse_uint32(num_labels);

    u_int8_t* scanned_image = malloc(sizeof(u_int8_t)*rows*cols);
    for (int i = 0; i < 10000; i++) //lets test so many images first
    {
        fread(scanned_image, 1, rows*cols, images);
        fread(&expected_number, 1, 1, label);
        for (int j = 0; j < rows; j++)
        {
            for (int k = 0; k < cols; k++)
            {
                image[j][k] = (double)scanned_image[j*cols + k]/255;
            }
        }
        feedforward(neural_net, image);
        back_propagate(neural_net, expected_number);
    }
    free(scanned_image);
    fclose(images);
    fclose(label);
}

void initialize_random(NeuralNet* neural_net)
{
    for (int i = 0; i < neural_net->connection1_2->size; i++)
    {
        neural_net->connection1_2->connectionArray[i]->weight = rand_uniform(-0.5f, 0.5f);
    }
    for (int i = 0; i < neural_net->connection2_3->size; i++)
    {
        neural_net->connection2_3->connectionArray[i]->weight = rand_uniform(-0.5f, 0.5f);
    }
    for (int i = 0; i < neural_net->connection3_4->size; i++)
    {
        neural_net->connection3_4->connectionArray[i]->weight = rand_uniform(-0.5f, 0.5f);
    }

    for (int i = 0; i < neural_net->layer4->size; i++)
    {
        neural_net->layer4->neuronArray[i]->bias = rand_uniform(-0.5f, 0.5f);
    }
    for (int i = 0; i < neural_net->layer3->size; i++)
    {
        neural_net->layer3->neuronArray[i]->bias = rand_uniform(-0.5f, 0.5f);
    }
    for (int i = 0; i < neural_net->layer2->size; i++)
    {
        neural_net->layer2->neuronArray[i]->bias = rand_uniform(-0.5f, 0.5f);
    }
}