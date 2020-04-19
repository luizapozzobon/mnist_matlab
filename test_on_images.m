clc; clear all; close all;

% 1º Teste das imagens de validação do dataset
% 2º Teste em imagens geradas por mim

%% Carregando dataset mnist
d = load('mnist.mat');
x_test = double(d.trainX)' ./ 255.0;
y_test = d.trainY;

%% Carregando modelo pré-treinado
trained = load('resultados/model_mnist_full');
model = trained.model;

%% Resultados de Validação - MNIST

% 10 amostras aleatórias de 1 a 10000 (número de imagens de teste disponíveis)
n_samples = 10
r = randi([1 10000], 1, n_samples);

figure(1)
for index=1:n_samples
    % Passagem forward no modelo para obtenção da predição
    img = x_test(:, index);
    output = model.forward(img);
    soft_pred = softmax(output);
    [confidences, pred] = max(soft_pred);
    pred = pred(1, :) - 1;
    
    % Reshape na imagem para plot -> [784, 1] -> [28, 28]
    img = fliplr(rot90(reshape(img, 28, 28), 3));
    label = y_test(index);
    
    subplot(2, 5, index)
    imshow(img)
    title(sprintf('Rótulo: %d \n Predição: %d', label, pred))
end

%% Resultado Testes com dígitos escritos por nós

labels = [0 1 2 5 8 22]; % nomes dos arquivos das imagens
show = 4; % índice da lista acima para mostrar o pré-processamento

for i=0:length(labels) - 1
    % Mudança do threshold de binarização para cada tipo de imagem
    % i <= 3 são as imagens da folha de papel que eu fiz
    % i > 3 são as escritas no quadro durante a apresentação
    if i <= 3
       threshold = 0.3;
    else
       threshold = 0.7;
    end
    
    figure(2)
    % Abrir imagem
    test_image = imread(sprintf('teste/%d.jfif', labels(i+1)));
    if i == show
        subplot(1, 4, 1)
        imshow(test_image)
        title('Imagem original')
    end
    
    % Redimensionar para 28x28 e Binarizar
    test_image = im2bw(imresize(test_image, [28, 28]), threshold);
    if i == show
        subplot(1, 4, 2)
        imshow(test_image)
        title('Após binarização')
    end
    
    % Inverter branco e preto
    test_image = ~test_image;
    if i == show
        subplot(1, 4, 3)
        imshow(test_image)
        title(sprintf('Inversão \n de cores'))
    end
    % Aplicar as mesmas rotações do dataset
    test_flipped = fliplr(rot90(test_image, 3));
    if i == show
        subplot(1, 4, 4)
        imshow(test_flipped)
        title(sprintf('Rotação 270º \n e flip vertical'))
    end
    
    % Formato de entrada da rede -> [28, 28] -> [784, 1]
    test_flat = reshape(test_flipped, [], 1);

    % Passagem forward para obtenção da predição
    output = model.forward(test_flat);
    soft_pred = softmax(output);
    [confidences, pred] = max(soft_pred);
    pred = pred(1, :) - 1; % predição

    figure(3)
    subplot(1, length(labels), i+1);
    imshow(test_image);
    title(sprintf("Predição: %d", pred))
    hold on;
end