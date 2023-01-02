{
    let random = function(){
        return (Math.random() - 0.5) * 2;
    }
    let relu = function(num){
        if(num > 0)
            return num;
        else
            return 0;
    }
    let sigmoid = function(num){
        return 1 / (1 + Math.exp(-num));
    }
    let squish = function(num){
        if(!num && num !== 0)
            throw new Error("The number provided is not real.");
        return sigmoid(num);
    }
    let round = function(num){
        return Math.round(num * 100) / 100;
    }
    let time = function(){
        return performance.now();
    }
    class Node{
        constructor(id){
            this.activation = 0;
            this.value = 0;
            this.bias = 0;
            this.target = 0;
            this.gradient = 0;
            this.weight_changes = [];
            this.weights = [];
            this.id = id;
        }
    }
    class NeuralNetwork{
        constructor(layout, epoch, learning_rate){
            if(layout.length < 2)
                throw new Error("The network layout is too small.");
        
            if(typeof layout !== "object")
                throw new Error("The network layout must be an array of numbers.");

            if(typeof epoch !== "number")
                throw new Error("The epoch must be a number.");

            if(typeof learning_rate !== "number")
                this.learning_rate = 0.5;
            else
                this.learning_rate = learning_rate;
            
            this.layers = [];
            this.layout = layout;
            this.epoch = epoch;

            this.cost = 0;
            this.average_cost = 0;
            this.total_average_cost = 0;

            this.runs = 0;
            this.time = 0;
            this.total_time = 0;

            //Intialize layers
            for(let i = 0; i < layout.length; i++){
                let layer_size = layout[i];
                this.layers.push([]);
                for(let l = 0; l < layer_size; l++){
                    let node = new Node(l);
                    if(i > 0){
                        for(let x = 0; x < layout[i - 1]; x++){
                            node.weight_changes.push(0);
                            node.weights.push(random());
                        }
                    }
                    this.layers[i].push(node);
                }
            }
        }
        run(inputs){
            if(inputs.length !== this.layers[0].length)
                throw new Error("The inputs provided and the input layer are different sizes. (Inputs: " + inputs.length + ", Input Layer: " + this.layers[0].length + ")");
            
            //Set first layers as inputs
            for(let i = 0; i < inputs.length; i++){
                let input = inputs[i];
                let node = this.layers[0][i];
                node.value = input;
                node.activation = input;
            }
            // console.log(this.layers[0]);

            //Perform neural network calculation
            for(let i = 1; i < this.layers.length; i++){
                let layer = this.layers[i];
                let previous_layer = this.layers[i - 1];
                for(let l = 0; l < layer.length; l++){
                    let node = layer[l];
                    node.id = l;
                    node.value = 0;
                    for(let x = 0; x < previous_layer.length; x++)
                        node.value += previous_layer[x].activation * node.weights[x];
                    node.value += node.bias;
                    node.activation = squish(node.value);
                }
            }

            this.runs++;
            return this.layers[this.layers.length - 1];
        }
        print_debug(){
            console.log("Run " + this.runs);
            let output_layer = this.layers[this.layers.length - 1];
            for(let i = 0; i < output_layer.length; i++)
                console.log("Output " + i + ": " + output_layer[i].activation);
            console.log("Cost: " + this.cost, "Average: " + this.average_cost, "Total Average: " + this.total_average_cost);
        }
        learn(targets){
            let output_layer = this.layers[this.layers.length - 1];
            if(targets.length !== output_layer.length)
                throw new Error("An incorrect target length was provided. The network output length is " + output_layer.length + ", but the targets provided have a length of " + targets.length);

            //Calculate gradients and changes for the input layer and the one after it
            this.cost = 0;
            for(let i = 0; i < output_layer.length; i++){
                let node = output_layer[i];
                node.gradient = (node.activation - targets[i]) * (node.activation * (1 - node.activation));
                this.cost += Math.pow(node.activation - targets[i], 2);
                for(let l = 0; l < node.weights.length; l++){
                    let previous_node = this.layers[this.layers.length - 2][l];
                    node.weight_changes[l] -= this.learning_rate * (node.gradient * previous_node.activation);
                }
            }
            this.average_cost = (this.cost + this.epoch * this.average_cost) / (this.epoch + 1);
            this.total_average_cost = (this.cost + this.runs * this.total_average_cost) / (this.runs + 1);

            //Update the rest of the hidden layers
            for(let i = this.layers.length - 2; i > 0; i--){
                let layer = this.layers[i];
                let previous_layer = this.layers[i - 1];
                for(let l = 0; l < layer.length; l++){
                    let node = layer[l];
                    node.gradient = this.sum_gradient(l, i + 1) * (node.activation * (1 - node.activation));
                    for(let x = 0; x < node.weights.length; x++)
                        node.weight_changes[x] -= this.learning_rate * (node.gradient * previous_layer[x].activation)
                }
            }

            // //Apply the weight changes after an epoch passes
            if((this.runs - 1) % this.epoch === this.epoch - 1){
                for(let i = 0; i < this.layers.length; i++){
                    for(let l = 0; l < this.layers[i].length; l++){
                        let node = this.layers[i][l];
                        // node.bias += this.learning_rate * node.gradient;
                        node.gradient = 0;
                        for(let x = 0; x < node.weight_changes.length; x++){
                            // console.log(node.weight_changes[x] / this.epoch, "Connection: " + (x + 1), "Node: " + (l + 1), "Layer: " + (i + 1));
                            node.weights[x] += node.weight_changes[x] / this.epoch;//Get average change in weights
                            node.weight_changes[x] = 0;
                        }
                    }
                }
            }
        }
        sum_gradient(node_index, layer_index){
            let result = 0;
            let layer = this.layers[layer_index];
            for(let i = 0; i < layer.length; i++){
                let node = layer[i];
                result += node.weights[node_index] * node.gradient;
            }
            return result;
        }
        train(inputs, targets, print_output){
            let output = this.run(inputs);
            this.learn(targets);
            if(print_output)
                this.print_debug();
            return output
        }
    }
    function create_network(layout, epoch, learning_rate){
        return new NeuralNetwork(layout, epoch, learning_rate);
    }
    function export_network(network){
        let result = {
            layout: network.layout,
            layers: []
        }
    }
}