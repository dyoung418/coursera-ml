function [theta, J_history] = runMultipleGD(X, y, theta, alphas, num_iters)

colors = ['k'; 'r'; 'g'; 'b'; 'm'; 'c'; 'y'];
styles = ['-'; '--'; ':'; "-."];
color_style_indicie_combos = perms([length(colors), length(styles)]);

% Plot the convergence graph
figure;
%plot(zeros(1), '-b', 'LineWidth', 2, 'LineStyle', '-');
plot(zeros(1), '-b', 'LineWidth', 4, 'LineStyle', '-');
xlabel('Number of iterations');
ylabel('Cost J');

numGDs = length(alphas);
for runs = 1:numGDs,
    [newTheta, J_history] = gradientDescentMulti(X, y, theta, alphas(runs), num_iters(runs));
    % Display gradient descent's result
    %printf('Theta computed from gradient descent: \n');
    %printf(' %f \n', newTheta);
    %printf('\n');
    %printf('debug, runs=%d, color=%d, style=%d', runs, mod(runs, length(colors)), max([1, mod(floor(runs/length(styles)), length(styles))]))
    color = colors(max([1, mod(runs, length(colors))]));
    style = styles(max([1, mod(floor(runs/length(styles)), length(styles))]));
    plot(1:numel(J_history), J_history, strcat(style, color, ';', sprintf('%d',alphas(runs)), ';'), 'LineWidth', 3);
    hold on;
end
theta = newTheta;

end
