classdef Linear
    properties
        output_size
        input_size
        weights
        bias
        output
        activation
    end
    methods
        function obj = Linear(n_in, n_out, activation)
            % Inicializa��o da camada linear (totalmente conectada)
            obj.input_size = n_in; % Quantia de neuronios de entrada
            obj.output_size = n_out; % Quantia de neuronios de sa�da
            obj.activation = activation; % Fun��o de ativa��o - string ou 0
            
            % Z (sa�da) = (n_out, m)
            obj.output = zeros(n_out, 1);
            for out=1:n_out
                % Pesos definidos pelo tamanho de entrada da camada
                for in=1:n_in
                    % W (pesos) = (n_out, n_in)
                    obj.weights(out, in) = rand()/100; % Pesos baixos s�o melhores para a converg�ncia
                end
                % Bias para esse neur�nio
                % b = (n_out, m)
                obj.bias(n_out, 1) = 1;
            end
        end
    end
end