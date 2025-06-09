#include "app.h"
#include "neural_net.h"
int main(void)
{
    //to test this malloc and free shit where fast a memory corruption bug can occur

    NeuralNet* net = instantiate_neural_net();
    load_wb(net);
    initialize_random(net);
    train(net, FILEPATH_TRAIN_DATA_IMAGE, FILEPATH_TRAIN_DATA_LABEL);
    //printf("this is the new version\n");
    //printf("this is the new version\n");

    save_wb(net);
    close_neural_net(net);
    app();
    return 0;
}
