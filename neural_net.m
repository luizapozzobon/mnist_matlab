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

%% Cria��o do modelo
% Camadas lineares (totalmente conectadas) = Linear(input, output, ativa��o)
% com pesos aleat�rios entre 0 e 1 e bias sempre de valor 1.
% Se n�o quiser ativa��o, digitar 0 ao inv�s do 'relu'

% Sigmoid ainda n�o est� funcionando
%model = Model([Linear(2, 3, 'sigmoid'), Linear(3, 2, 'sigmoid'), Linear(2, 1, 'sigmoid')]);

%model = Model([Linear(2, 3, 'relu'), Linear(3, 2, 'relu'), Linear(2, 1, 'relu')]);
model = Model([Linear(784, 256, 'relu'), Linear(256, 256, 'relu'), Linear(256, 10, 'relu')]); % sem ativa��o

%% Hiperpar�metros
max_epochs = 25;
batch_size = 128;
lr = 0.03;

% Contadores
batch_count = 0;
val_batch_count = 0;

% Reduzir a learning rate em 10% nessas �pocas
epochs_milestones = [10, 20, 30];

% Se for usar como crit�rio o valor da loss
%epoch_loss = 20000;
%desired_loss = 1; % precis�o

%while epoch_loss > desired_loss
for epochs=1:max_epochs
    epochs
    % In�cio de uma �poca
    start_sample = 1;
    val_start_sample = 1;
    
    if ismember(epochs, epochs_milestones)
        lr = lr/10.0
    end
    
    for batch=1:length(targets)/batch_size
        % 128 samples dessa itera��o
        end_sample = batch * batch_size;
            
        % Sele��o das samples desse batch e One hot encode dos labels 
        target = to_categorical(targets(start_sample:end_sample), n_classes);
        input = inputs(:, start_sample:end_sample);

        % Forward propagation
        output = model.forward(input);
        
        % Softmax na sa�da da passagem forward do modelo
        soft_pred = softmax(output);
        % Index do maior valor ap�s softmax � o label previsto pela rede
        [confidences, pred] = max(soft_pred);
        
        % Salvando o resultado obtido
        % Subtrai 1 para ir de 1 ate 10 (indexes do array) de 0 at� 9 (labels corretos)
        pred = pred(1, :) - 1;

        % C�lculo do erro
        error = negative_log_likelihood(confidences);
        % Custo m�dio nesse batch
        cost = sum(error)/batch_size;
        
        if mod(batch, 100) == 0
            cost
        end
       
        % Backprop para atualiza��o dos pesos dos neur�nios
        model.backprop(input, soft_pred, target, lr, batch_size);
        
        % Registro dos erros dos batches de uma �poca
        batches_loss(batch) = cost;
        batch_count = batch_count + 1;
        start_sample = end_sample + 1;
    end
    train_mean_loss(epochs) = mean(batches_loss) % Loss de uma �poca (conjunto de n batches) -> crit�rio de parada
    
    %% ==== VALIDATION
    % Ap�s uma �poca de treino, a valida��o � realizada para obtermos a
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
        
        % C�lculo da acur�cia de valida��o
        accs(batch) = accuracy(val_targets(val_start_sample:val_end_sample), pred);

        % C�lculo do erro
        error = negative_log_likelihood(confidences);
        cost = sum(error)/batch_size;
        
        if mod(batch, 50) == 0
           cost
        end
        
        val_batches_loss(batch) = cost; % Registro dos erros dos batches de uma �poca
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
display('�pocas de treino')
display(epochs)
display('Loss da �ltima �poca: ')
display(val_mean_loss(end))
display('Acur�cia da �ltima �poca: ')
display(val_mean_acc(end))

figure(1)
plot(val_mean_acc)
title('Acur�cia m�dia das �pocas de valida��o')
ylabel('Acur�cia')
xlabel('�poca')

figure(2)
plot(val_mean_loss)
title('Erro (loss) m�dio das �pocas de valida��o')
ylabel('Erro (loss)')
xlabel('�poca')
