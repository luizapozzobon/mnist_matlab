clc; clear all; close all;

% 1� Teste das imagens de valida��o do dataset
% 2� Teste em imagens geradas por mim

%% Carregando dataset mnist
d = load('mnist.mat');
x_test = double(d.trainX)' ./ 255.0;
y_test = d.trainY;

%% Carregando modelo pr�-treinado
trained = load('resultados/model_mnist_full');
model = trained.model;

%% Resultados de Valida��o - MNIST

% 10 amostras aleat�rias de 1 a 10000 (n�mero de imagens de teste dispon�veis)
n_samples = 10
r = randi([1 10000], 1, n_samples);

figure(1)
for index=1:n_samples
    % Passagem forward no modelo para obten��o da predi��o
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
    title(sprintf('R�tulo: %d \n Predi��o: %d', label, pred))
end

%% Resultado Testes com d�gitos escritos por n�s

labels = [0 1 2 5 8 22]; % nomes dos arquivos das imagens
show = 4; % �ndice da lista acima para mostrar o pr�-processamento

for i=0:length(labels) - 1
    % Mudan�a do threshold de binariza��o para cada tipo de imagem
    % i <= 3 s�o as imagens da folha de papel que eu fiz
    % i > 3 s�o as escritas no quadro durante a apresenta��o
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
        title('Ap�s binariza��o')
    end
    
    % Inverter branco e preto
    test_image = ~test_image;
    if i == show
        subplot(1, 4, 3)
        imshow(test_image)
        title(sprintf('Invers�o \n de cores'))
    end
    % Aplicar as mesmas rota��es do dataset
    test_flipped = fliplr(rot90(test_image, 3));
    if i == show
        subplot(1, 4, 4)
        imshow(test_flipped)
        title(sprintf('Rota��o 270� \n e flip vertical'))
    end
    
    % Formato de entrada da rede -> [28, 28] -> [784, 1]
    test_flat = reshape(test_flipped, [], 1);

    % Passagem forward para obten��o da predi��o
    output = model.forward(test_flat);
    soft_pred = softmax(output);
    [confidences, pred] = max(soft_pred);
    pred = pred(1, :) - 1; % predi��o

    figure(3)
    subplot(1, length(labels), i+1);
    imshow(test_image);
    title(sprintf("Predi��o: %d", pred))
    hold on;
end