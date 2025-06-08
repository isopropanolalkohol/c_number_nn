//
// Created by joshua on 08.06.25.
//

#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#define LAYER_1 784
#define LAYER_2 64
#define LAYER_3 32
#define LAYER_4 10
#define FILEPATH "../assets/weights_biases.txt"

struct Connection;
struct Neuron;
struct NeuronLayer;
struct NeuralNetwork;
struct ConnectionLayer;

typedef struct Connection
{
    double weight;
    struct Neuron *source;
    struct Neuron *dest;

} Connection;

typedef struct Neuron
{
    double activation;
    double bias;
} Neuron;

typedef struct NeuronLayer
{
    Neuron** neuronArray;
    int size;
} NeuronLayer;

typedef struct ConnectionLayer
{
    Connection** connectionArray;
    int size;
} ConnectionLayer;


typedef struct NeuralNetwork
{
    NeuronLayer* layer1;
    ConnectionLayer* connection1_2;
    NeuronLayer* layer2;
    ConnectionLayer* connection2_3;
    NeuronLayer* layer3;
    ConnectionLayer* connection3_4;
    NeuronLayer* layer4;
} NeuralNet;

NeuralNet* instantiate_neural_net();
void load_wb(NeuralNet* neural_net, const char* filepath);
void save_wb(NeuralNet* neural_net, const char* filepath);
void close_neural_net(NeuralNet* neural_net);


ConnectionLayer* instantiate_connection_layer(NeuronLayer* layer1, NeuronLayer* layer2);
void free_connection_layer(ConnectionLayer* layer);

NeuronLayer* instantiate_neuron_layer(const int neuronCount);
void free_neuron_layer(NeuronLayer* layer);

#endif //NEURAL_NET_H