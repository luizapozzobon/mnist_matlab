clc; clear all; close all;

rng(42);

% https://www.mathworks.com/matlabcentral/fileexchange/69480-sample-deep-network-training-with-mnist-and-cifar
% fontes boas:
% https://towardsdatascience.com/neural-net-from-scratch-using-numpy-71a31f6e3675
% https://towardsdatascience.com/building-an-artificial-neural-network-using-pure-numpy-3fe21acc5815
% https://zhenye-na.github.io/2018/09/09/build-neural-network-with-mnist-from-scratch.html

% Neural Networks
% Luiza Pozzobon

%% Carregamento do dataset
d = load('mnist.mat');
x_train = double(d.trainX)';
y_train = d.trainY;
x_test = double(d.testX)';
y_test = d.testY;

y_train = y_train(1, :);
y_test = y_test(1, :);

classes = unique(y_train)
n_classes = length(classes)
x_train_shape = size(x_train)

targets = y_train;
inputs = x_train./255.0;
inputs(1,:);

% Limitar set de treino
%targets = targets(1, 1:5000);
%inputs = inputs(:, 1:5000);

val_targets = y_test;
val_inputs = x_test./255.0;

%i = reshape(x_train(1,:), 28, 28)';
%label = y_test(1)
%image(i);

%% Criação do modelo
% Camadas lineares (totalmente conectadas) = Linear(input, output, ativação)
% com pesos aleatórios entre 0 e 1 e bias sempre de valor 1.
% Se não quiser ativação, digitar 0 ao invés do 'relu'

% Sigmoid ainda não está funcionando
%model = Model([Linear(2, 3, 'sigmoid'), Linear(3, 2, 'sigmoid'), Linear(2, 1, 'sigmoid')]);

%model = Model([Linear(2, 3, 'relu'), Linear(3, 2, 'relu'), Linear(2, 1, 'relu')]);
model = Model([Linear(784, 256, 'relu'), Linear(256, 256, 'relu'), Linear(256, 10, 'relu')]); % sem ativação

%% Hiperparâmetros
max_epochs = 25;
batch_size = 128;
lr = 0.03;

% Contadores
batch_count = 0;
val_batch_count = 0;

% Reduzir a learning rate em 10% nessas épocas
epochs_milestones = [10, 20, 30];

% Se for usar como critério o valor da loss
%epoch_loss = 20000;
%desired_loss = 1; % precisão

%while epoch_loss > desired_loss
for epochs=1:max_epochs
    epochs
    % Início de uma época
    start_sample = 1;
    val_start_sample = 1;
    
    if ismember(epochs, epochs_milestones)
        lr = lr/10.0
    end
    
    for batch=1:length(targets)/batch_size
        % 128 samples dessa iteração
        end_sample = batch * batch_size;
            
        % Seleção das samples desse batch e One hot encode dos labels 
        target = to_categorical(targets(start_sample:end_sample), n_classes);
        input = inputs(:, start_sample:end_sample);

        % Forward propagation
        output = model.forward(input);
        
        % Softmax na saída da passagem forward do modelo
        soft_pred = softmax(output);
        % Index do maior valor após softmax é o label previsto pela rede
        [confidences, pred] = max(soft_pred);
        
        % Salvando o resultado obtido
        % Subtrai 1 para ir de 1 ate 10 (indexes do array) de 0 até 9 (labels corretos)
        pred = pred(1, :) - 1;

        % Cálculo do erro
        error = negative_log_likelihood(confidences);
        % Custo médio nesse batch
        cost = sum(error)/batch_size;
        
        if mod(batch, 100) == 0
            cost
        end
       
        % Backprop para atualização dos pesos dos neurônios
        model.backprop(input, soft_pred, target, lr, batch_size);
        
        % Registro dos erros dos batches de uma época
        batches_loss(batch) = cost;
        batch_count = batch_count + 1;
        start_sample = end_sample + 1;
    end
    train_mean_loss(epochs) = mean(batches_loss) % Loss de uma época (conjunto de n batches) -> critério de parada
    
    %% ==== VALIDATION
    % Após uma época de treino, a validação é realizada para obtermos a
    % performance atual do modelo em dados nunca vistos pela rede.
    for batch=1:length(val_targets)/batch_size
        val_end_sample = batch * batch_size;
        
        target = to_categorical(val_targets(val_start_sample:val_end_sample), n_classes);
        input = val_inputs(:, val_start_sample:val_end_sample);

        % Forward propagation
        output = model.forward(input);
        % Softmax
        soft_pred = softmax(output);
        [confidences, pred] = max(soft_pred);
        
        % Salvando o resultado obtido
        pred = pred(1, :) - 1;
        
        % Cálculo da acurácia de validação
        accs(batch) = accuracy(val_targets(val_start_sample:val_end_sample), pred);

        % Cálculo do erro
        error = negative_log_likelihood(confidences);
        cost = sum(error)/batch_size;
        
        if mod(batch, 50) == 0
           cost
        end
        
        val_batches_loss(batch) = cost; % Registro dos erros dos batches de uma época
        val_batch_count = val_batch_count + 1;
        val_start_sample = val_end_sample + 1;
    end
    val_mean_loss(epochs) = mean(val_batches_loss);
    val_mean_acc(epochs) = mean(accs);
    %epochs = epochs + 1
end

% save('model_mnist_5k', 'model')

display('Batches treinados: ')
display(batch_count)
display('Épocas de treino')
display(epochs)
display('Loss da última época: ')
display(val_mean_loss(end))
display('Acurácia da última época: ')
display(val_mean_acc(end))

figure(1)
plot(val_mean_acc)
title('Acurácia média das épocas de validação')
ylabel('Acurácia')
xlabel('Época')

figure(2)
plot(val_mean_loss)
title('Erro (loss) médio das épocas de validação')
ylabel('Erro (loss)')
xlabel('Época')
