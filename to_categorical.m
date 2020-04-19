function [one_hot] = to_categorical(numeric_label, n_classes)
    for sample=1:size(numeric_label, 2)
        one_hot(:, sample) = zeros(n_classes, 1);
        one_hot(numeric_label(1, sample) + 1, sample) = 1;
    end
end