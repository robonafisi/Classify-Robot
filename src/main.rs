[dependencies]
burn = "0.1"  
burn-tensor = "0.1"  
rand = "0.8"  


use burn::tensor::Tensor;
use burn::nn::{Linear, ReLU, Dropout, Sequential, Module, CrossEntropyLoss};
use burn::optim::{SGD, Optimizer};
use rand::seq::SliceRandom;
use rand::thread_rng;


struct MLP {
    layers: Sequential,
}

impl MLP {
    fn new(input_dim: usize, hidden_dims: &[usize], output_dim: usize) -> Self {
        let mut layers = Sequential::new(vec![]);
        let mut prev_dim = input_dim;

        for &dim in hidden_dims {
            layers.push(Box::new(Linear::new(prev_dim, dim)));
            layers.push(Box::new(ReLU::new()));
            layers.push(Box::new(Dropout::new(0.5)));
            prev_dim = dim;
        }

        layers.push(Box::new(Linear::new(prev_dim, output_dim)));
        Self { layers }
    }
}

impl Module for MLP {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.layers.forward(x)
    }
}


//Create synthetic data to run an example
fn generate_data(num_samples: usize, input_dim: usize) -> (Tensor, Tensor) {
    let mut rng = thread_rng();
    let inputs: Vec<Vec<f64>> = (0..num_samples)
        .map(|_| (0..input_dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    let targets: Vec<u64> = inputs
        .iter()
        .map(|x| {
            if x[0] > 0.5 {
                0 
            } else if x[1] > 0.5 {
                1 
            } else {
                2 
            }
        })
        .collect();

    let inputs_tensor = Tensor::from(inputs);
    let targets_tensor = Tensor::from(targets);
    (inputs_tensor, targets_tensor)
}


// Prepare the data for the model
fn split_data(inputs: &Tensor, targets: &Tensor, split_ratio: f64) -> ((Tensor, Tensor), (Tensor, Tensor)) {
    let num_samples = inputs.shape()[0];
    let mut indices: Vec<usize> = (0..num_samples).collect();
    let mut rng = thread_rng();
    indices.shuffle(&mut rng);

    let split_index = (num_samples as f64 * split_ratio).round() as usize;
    let (train_indices, val_indices) = indices.split_at(split_index);

    let train_inputs = Tensor::from(train_indices.iter().map(|&i| inputs[i].clone()).collect::<Vec<_>>());
    let train_targets = Tensor::from(train_indices.iter().map(|&i| targets[i].clone()).collect::<Vec<_>>());
    let val_inputs = Tensor::from(val_indices.iter().map(|&i| inputs[i].clone()).collect::<Vec<_>>());
    let val_targets = Tensor::from(val_indices.iter().map(|&i| targets[i].clone()).collect::<Vec<_>>());

    ((train_inputs, train_targets), (val_inputs, val_targets))
}

//Parameters for the ML model
fn train_model(
    model: &mut MLP,
    train_inputs: &Tensor,
    train_targets: &Tensor,
    val_inputs: &Tensor,
    val_targets: &Tensor,
    epochs: usize,
    learning_rate: f64,
) {
    let mut optimizer = SGD::new(learning_rate);

    for epoch in 0..epochs {
 
        let predictions = model.forward(train_inputs);

  
        let loss = CrossEntropyLoss::forward(&predictions, train_targets);
        println!("Epoch {}: Training Loss = {:?}", epoch, loss);

    
        model.zero_grad();
        loss.backward();


        optimizer.step(model);


        let val_predictions = model.forward(val_inputs);
        let val_loss = CrossEntropyLoss::forward(&val_predictions, val_targets);
        let correct_predictions = val_predictions
            .argmax(1)
            .eq(val_targets)
            .sum()
            .to_scalar::<f64>();
        let accuracy = correct_predictions / val_inputs.shape()[0] as f64;

        println!("Epoch {}: Validation Loss = {:?}, Accuracy = {:.2}%", epoch, val_loss, accuracy * 100.0);
    }
}


fn main() {

    let input_dim = 10;  
    let hidden_dims = [64, 32];
    let output_dim = 3;  
    let num_samples = 1000;
    let split_ratio = 0.8;
    let epochs = 50;
    let learning_rate = 0.01;

    let (inputs, targets) = generate_data(num_samples, input_dim);
    let ((train_inputs, train_targets), (val_inputs, val_targets)) = split_data(&inputs, &targets, split_ratio);


    let mut model = MLP::new(input_dim, &hidden_dims, output_dim);
    train_model(&mut model, &train_inputs, &train_targets, &val_inputs, &val_targets, epochs, learning_rate);
}
