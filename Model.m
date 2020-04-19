classdef Model < handle
    properties
        layers
    end
    methods
        function obj = Model(layers)
            obj.layers = layers;
        end
        
        function [A] = forward(obj, input)
            % Passagem forward da rede neural
            % Itera por todas as camadas definidas na cria��o da classe Model
            A = input;
            
            % m = batch_size
            % A = (n_out, m)
            % Z = (n_out, m)
            % b = (n_out, m)
            % W = (n_out, n_in)
            
            for l=1:length(obj.layers)
                % output = Weights*input + bias
                W = obj.layers(l).weights;
                bias = obj.layers(l).bias;
                
                Z = (W * A) + bias;
                
                % Fun��o de ativa��o na sa�da da camada linear, se houver
                if obj.layers(l).activation ~= 0
                    A = activation(Z, obj.layers(l).activation);
                else
                    A = Z;
                end
                
                obj.layers(l).output = A;
                
                if isnan(A)
                    A
                end
            end
        end
        
        function [] = backprop(obj, input, pred, target, lr, m)
            
            % Erro da �ltima camada - predi��o e label esperado
            dZ2 = pred - target;
            
            % Atualiza��o dos pesos da camada 'n' at� a 2
            for l=length(obj.layers):-1:2
                % Peso da camada atual
                W2 = obj.layers(l).weights;
                % Ativa��o (sa�da) da camada anterior
                A1 = obj.layers(l-1).output;
                
                % Gradiente dos pesos e bias da camada atual
                [dW, db] = obj.layer_backprop(dZ2, A1, m);
                % Atualiza��o dos pesos a partir dos gradientes obtidos
                obj.layer_update(l, dW, db, lr);

                % C�lculo do novo erro para a pr�xima camada
                dZ2 = W2' * dZ2;
            end 
            
            % Atualiza��o dos pesos da primeira camada
            [dW, db] = obj.layer_backprop(dZ2, input, m);
            obj.layer_update(1, dW, db, lr);
        end
        
        function [dW, db] = layer_backprop(obj, dZ, A, m)
            % m  = batch_size
            % dZ = (n_out, m)
            % A' = (m, n_in)
            % dW = (n_out, n_in)
            % db = (n_out, m)
            dW = 1/m * dZ * A';
            db = 1/m * sum(dZ, 2);
        end
        
        function [] = layer_update(obj, index, dW, db, lr)
            % Atualiza��o dos pesos e bias
            W = obj.layers(index).weights;
            b = obj.layers(index).bias;
            
            obj.layers(index).weights = W - lr * dW;
            obj.layers(index).bias = b - lr * db;
        end
    end
end