function [acc] = accuracy(target, prediction)
    correct = 0;
    for i=1:length(target)
        if target(i) == prediction(i)
            correct = correct + 1;
        end
    end
    acc = correct/length(target)
    target(15:30)
    prediction(15:30)
end