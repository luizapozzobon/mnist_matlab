function [activated] = activation(outputs, activation)
% Aplicação de funções de ativação em um determinado conjunto de dados
% (outputs)

% https://www.mathworks.com/matlabcentral/fileexchange/59223-convolution-neural-network-simple-code-simple-to-use

    if activation == "relu"
        indexes = (outputs > 0);
        activated = indexes .* outputs;
    elseif activation == "sigmoid"
        sig = @(x) 1/(1 + exp(-x));
        for i=1:length(outputs)
            activated(i) = sig(outputs(i));
        end
    end
end