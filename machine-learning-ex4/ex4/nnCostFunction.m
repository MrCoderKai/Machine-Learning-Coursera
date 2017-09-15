function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% add bias unit to X
X = [ones(m, 1) X];

y_m = repmat((1:num_labels)', 1, m);

% a stupid way to compute y_m where y_m is a matrix and each coloum
% corresponds to a scalar in vector y.
% for i = 1:m
%     y_m(:, i) = y_m(:, i) == y(i);
% end

% a fast way to compute y_m
y_m = bsxfun(@eq, y_m, y');

% hidden units
a2 = sigmoid(Theta1 * X');
% add bias unit to hidden units
a2 = [ones(1, m); a2];
% output units (hypothesis)
a3 = sigmoid(Theta2 * a2);

J = 1.0 / m * sum(sum(-y_m .* log(a3) - (1 - y_m) .* ...
    log(1 - a3)));

% index of bias units for each layer in nn_params
index_bias_units = [1:hidden_layer_size ...
    1+(hidden_layer_size*(input_layer_size + 1)): ...
      (hidden_layer_size*(input_layer_size + 1))+num_labels];
% delete bias unit's parameters in nn_params
nn_params(index_bias_units) = []; 
% compute regularization term in regularized nerual network cost function
regularization_term = lambda / (2 * m) * (nn_params' * nn_params);

% add regularization term to nerual network cost function
J = J + regularization_term;

%% computation for gradient

% delta3: the difference between hypothesis and labels
delta3 = a3 - y_m;

z2 = [ones(1, m); Theta1 * X'];
delta2 = Theta2' * delta3 .* sigmoidGradient(z2);
% delete bias unit delta
delta2(1, :) = [];

Theta1_grad = delta2 * X / m;
Theta2_grad = delta3 * a2' / m;

% regularization term for gradient
Theta_grad_reg_term1 = zeros(size(Theta1_grad));
Theta_grad_reg_term2 = zeros(size(Theta2_grad));
Theta_grad_reg_term1(:, 1) = zeros(hidden_layer_size, 1);
Theta_grad_reg_term2(:, 1) = zeros(num_labels, 1);
Theta_grad_reg_term1(:, 2:end) = lambda / m * Theta1(:, 2:end);
Theta_grad_reg_term2(:, 2:end) = lambda / m * Theta2(:, 2:end);

% add regularization term to gradient
Theta1_grad = Theta1_grad + Theta_grad_reg_term1;
Theta2_grad = Theta2_grad + Theta_grad_reg_term2;












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
