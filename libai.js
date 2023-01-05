{
    let sigmoid = function(input){
        return 1 / (1 + Math.pow(Math.E, -input));
    }
    let random_range = 1;
    let random = function(){
        return (Math.random - 0.5) * 2 *random_range
    }
    let square = function(input){
        return input * input;
    }
    class Node{
        value = 0;
        gradient = 0;

        bias = 0;
        bias_change = 0;
        
        weights = [];
        changes = [];
        
        constructor(previous_layer_size){
            for(let i = 0; i < previous_layer_size; i++){
                weights[i] = random();
                changes[i] = 0;
            }
        }
    }

    class Network{
        layers = [];
        cost = 0;
        average_cost = 0;
        runs = 0;
        constructor(layout, epoch, learning_rate){
            
            //Init variables
            this.layout = layout;
            this.epoch = epoch;
            this.learning_rate = learning_rate;

            //Initialize layers
            for(let i = 0; i < layout.length; i++){
                layers[i] = [];
                for(let j = 0; j < layout[i]; j++){
                    if(i > 0)
                        layers[i][j] = new Node(layout[i - 1]);
                    else
                        layers[i][j] = new Node(0);
                }
            }
        }
        run(inputs){
            //Set first layer to inputs
            for(let i = 0; i < inputs.length; i++)
                layers[0][i].value = inputs[i];
            
            //Calculate network output
            for(let i = 1; i < this.layers.length; i++){
                for(let j = 0; j < this.layers[i].length; j++){
                    let node = this.layers[i][j];
                    node.value = 0;
                    for(let k = 0; k < node.weights.length; k++)
                        node.value += layers[i - 1][k].value * node.weights[k];
                    node.value += node.bias;
                    node.value = sigmoid(node.value);
                }
            }
        }
        get_cost(targets){
            this.cost = 0;
            for(let i = 0; i < targets.length; i++)
                this.cost += square(this.layers[this.layers.length - 1][i].value - targets[i]);
            this.average_cost = (this.cost + this.epoch * this.average_cost) / (this.epoch + 1);
        }
        learn(targets){
            //Get changes and gradients for output layer and the layer just after it
            for(let i = 0; i < targets.length; i++){
                let node = this.layers[this.layers.length - 1][i];
                node.gradient = (2 * (node.value - targets[i])) * (node.value * (1 - node.value));
                node.bias_change -= this.learning_rate * node.gradient;
                for(let j = 0; j < node.weights.length; j++)
                    node.changes[j] -= this.learning_rate * (node.gradient * this.layers[this.layers.length - 2][j].value);
            }

            //Calculate network cost
            this.get_cost(targets);

            //Update the rest of the layers
            for(let i = layers.length - 2; i > 0; i--){
                for(let j = 0; j < layers[i].length; j++){
                    let node = layers[i][j];
                    node.gradient = this.sum_gradient(j, i + 1) * (node.value * (1 - node.value));
                    node.bias_change -= this.learning_rate * node.gradient;
                    for(let k = 0; k < node.weights.length; k++)
                        node.changes[k] -= this.learning_rate * (node.gradient * this.layers[i - 1][k].value);
                }
            }

            //Update the weights and biases after every epoch
            if(this.runs % (this.epoch + 1) === this.epoch){
                for(let i = 0; i < this.layers.length; i++){
                    for(let j = 0; j < this.layers[i].length; j++){
                        let node = this.layers[i][j];
                        //Apply bias change
                        node.bias += node.bias_change;
                        node.bias_change = 0;
                        for(let k = 0; k < node.changes.length; k++){
                            //Apply weight changes
                            node.weights[k] += node.changes[k];
                            node.changes[k] = 0;
                        }
                    }
                }
            }
            this.runs++;
        }
        sum_gradient(node_index, layer_index){
            let result = 0;
            for(let i = 0; i < layers[node_index].length; i++)
                result += node.weights[node_index] * node.gradient;
            return result;
        }
        train(inputs, targets){
            this.run(inputs);
            this.learn(targets);
        }
        print_output(){
            for(let i = 0; i < this.layers[this.layers.length - 1].length; i++)
                console.log(this.layers[this.layers.length - 1][i].value);
        }
    }
}