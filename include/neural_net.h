//
// Created by joshua on 08.06.25.
//

#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#define LAYER_1 784
#define LAYER_2 64
#define LAYER_3 32
#define LAYER_4 10

#define ETA 0.3

#define FILEPATH_CONNLAYER1 "../assets/weights_biases/weights_connlayer1.txt"
#define FILEPATH_CONNLAYER2 "../assets/weights_biases/weights_connlayer2.txt"
#define FILEPATH_CONNLAYER3 "../assets/weights_biases/weights_connlayer3.txt"

// NEUR_LAYER1 is used for input
#define FILEPATH_NEURLAYER2 "../assets/weights_biases/biases_neurlayer2.txt"
#define FILEPATH_NEURLAYER3 "../assets/weights_biases/biases_neurlayer3.txt"
#define FILEPATH_NEURLAYER4 "../assets/weights_biases/biases_neurlayer4.txt"

#define FILEPATH_TRAIN_DATA_IMAGE "../assets/training_data/train-images.idx3-ubyte"
#define FILEPATH_TRAIN_DATA_LABEL "../assets/training_data/train-labels.idx1-ubyte"

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
    int input_size;
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
void load_wb(NeuralNet* neural_net);
void save_wb(NeuralNet* neural_net);
void close_neural_net(NeuralNet* neural_net);


ConnectionLayer* instantiate_connection_layer(NeuronLayer* layer1, NeuronLayer* layer2);
void free_connection_layer(ConnectionLayer* layer);

NeuronLayer* instantiate_neuron_layer(const int neuronCount);
void free_neuron_layer(NeuronLayer* layer);


void write_weights(const char* filepath, ConnectionLayer* layer);
void write_biases(const char* filepath, NeuronLayer* layer);

void read_weights(const char* filepath, ConnectionLayer* layer);
void read_biases(const char* filepath, const NeuronLayer* layer);

int feedforward(NeuralNet* neural_net, double* input);
void propagate(NeuronLayer* lhs, ConnectionLayer* conn,  NeuronLayer* rhs);

void back_propagate(const NeuralNet* neural_net, int expected_value);

double cost_function(const NeuralNet* neural_net, int expected_value);

typedef struct gradCf
{
    double* gradWeights_l1;
    double* gradWeights_l2;
    double* gradWeights_l3;
    double* gradBiases_l1;
    double* gradBiases_l2;
    double* gradBiases_l3;
} gradCf;

void initialize_random(NeuralNet* neural_net);
void train(NeuralNet* neural_net, const char* filepath_image, const char* filepath_label);


#endif //NEURAL_NET_H