function soft_out = softmax(input)
    for b=1:size(input, 2) % every sample from batch
        for i=1:size(input, 1)
            exps(i, b) = exp(input(i, b));
        end
        soma = sum(exps(:, b));
        soft_out(:, b) = exps(:, b) ./ soma;
    end
end