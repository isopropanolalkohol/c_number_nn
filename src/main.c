#include "app.h"
#include "neural_net.h"

int main(void)
{
    //start_app();

    NeuralNet* net = instantiate_neural_net();
    close_neural_net(net);
    return 0;
}
