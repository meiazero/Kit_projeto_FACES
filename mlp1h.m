function [STATS TX_OK W1 W2] = mlp1h(D, Nr, Ptrain)
  [N, p1] = size(D);
  p = p1 - 1;
  K = max(D(:,end));
  H = 20;
  TX_OK = zeros(Nr,1);

  for r=1:Nr
    idx = randperm(N);
    Dsh = D(idx,:);
    Ntrain = round(Ptrain/100 * N);
    Train = Dsh(1:Ntrain,:);
    Test = Dsh(Ntrain+1:end,:);

    Xtrain_raw = Train(:,1:p);
    m = mean(Xtrain_raw);
    s = std(Xtrain_raw);

    % Normalize features (z-score)
    Xtest_raw = Test(:,1:p);
    Xtrain = (Xtrain_raw - m) ./ s;
    Xtest = (Xtest_raw - m) ./ s;

    % Normalize features (mudanca de escala [0,+1])
    % Xtrain = (Xtrain_raw - min(Xtrain_raw)) ./ (max(Xtrain_raw) - min(Xtrain_raw));
    % Xtest = (Xtest_raw - min(Xtrain_raw)) ./ (max(Xtrain_raw) - min(Xtrain_raw));

    % Normalize features (mudanca de escala [-1,+1])
    % Xtrain = 2 * (Xtrain_raw - min(Xtrain_raw)) ./ (max(Xtrain_raw) - min(Xtrain_raw)) - 1;
    % Xtest = 2 * (Xtest_raw - min(Xtrain_raw)) ./ (max(Xtrain_raw) - min(Xtrain_raw)) - 1;

    % One-hot encode labels
    Ytrain_oh = zeros(Ntrain, K);
    for i=1:Ntrain
      Ytrain_oh(i, Train(i,end)) = 1;
    end

    % Initialize weights (He initialization for sigmoid)
    W1 = randn(p, H) * sqrt(2/p);
    b1 = zeros(1, H);
    W2 = randn(H, K) * sqrt(2/H);
    b2 = zeros(1, K);
    eta = 0.01; epochs = 200;
    losses = zeros(epochs,1);

    % Training loop
    for e=1:epochs
      Z1 = Xtrain * W1 + b1;

      % Sigmoid activation
      A1 = 1 ./ (1 + exp(-Z1));

      % Tanh activation
      % A1 = tanh(Z1);

      % ReLU activation
      % A1 = max(Z1, 0);

      % Leaky ReLU activation
      % A1 = max(Z1, 0) + 0.01 * Z1;

      % ReLU6 activation
      % A1 = min(max(Z1, 0), 6);

      Z2 = A1 * W2 + b2;
      A2 = softmax(Z2')';

      % Compute cross-entropy loss
      loss = -sum(sum(Ytrain_oh .* log(A2 + 1e-10))) / Ntrain;
      losses(e) = loss;

      % Backpropagation
      dZ2 = A2 - Ytrain_oh;
      dW2 = A1' * dZ2 / Ntrain;
      db2 = sum(dZ2) / Ntrain;
      dA1 = dZ2 * W2';

      % Sigmoid derivative
      dZ1 = dA1 .* A1 .* (1 - A1);

      % Tanh derivative
      % dZ1 = dA1 .* (1 - A1.^2);

      % ReLU derivative
      % dZ1 = dA1 .* (A1 > 0);

      % Leaky ReLU derivative
      % dZ1 = dA1 .* (A1 > 0) + dA1 .* (A1 <= 0) .* 0.01;

      % ReLU6 derivative
      % dZ1 = dA1 .* (A1 > 0) .* (A1 < 6);

      dW1 = Xtrain' * dZ1 / Ntrain;
      db1 = sum(dZ1) / Ntrain;
      W1 = W1 - eta * dW1;
      b1 = b1 - eta * db1;
      W2 = W2 - eta * dW2;
      b2 = b2 - eta * db2;

      % Log every 10 epochs
      % if mod(e, 10) == 0
      %   disp(["<MLP-1H, round = " num2str(r) ", epoch = " num2str(e) "/" num2str(epochs) ", loss = " num2str(loss) ">"]);
      % end
    end

    Z1_test = Xtest * W1 + b1;
    A1_test = 1 ./ (1 + exp(-Z1_test));
    Z2_test = A1_test * W2 + b2;
    A2_test = softmax(Z2_test')';
    [~, pred] = max(A2_test, [], 2);
    correct = sum(pred == Test(:,end));
    TX_OK(r) = correct / size(Test,1) * 100;

    % Log progress
    current_mean = mean(TX_OK(1:r));
    current_std = std(TX_OK(1:r));
    % disp(["<MLP-1H, round = " num2str(r) "/" num2str(Nr) ", mean acc = " num2str(current_mean) ", std acc = " num2str(current_std) ", final loss = " num2str(losses(end)) ">"]);

    % Plot loss for this round if last
    % if r == Nr
    %   figure;
    %   plot(1:epochs, losses, 'g-', 'LineWidth', 2);
    %   title('Training Loss over Epochs for MLP-1H (Last Round)', 'FontSize', 14, 'FontWeight', 'bold');
    %   xlabel('Epoch', 'FontSize', 12);
    %   ylabel('Cross-Entropy Loss', 'FontSize', 12);
    %   grid on;
    %   set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5);
    %   print -dpng 'mlp1h_loss_plot.png';
    % end
  end

  STATS = [mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];

  % Plot accuracy over rounds
  % figure;
  % plot(1:Nr, TX_OK, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
  % title('Accuracy over Rounds for MLP-1H', 'FontSize', 14, 'FontWeight', 'bold');
  % xlabel('Round Number', 'FontSize', 12);
  % ylabel('Test Accuracy (%)', 'FontSize', 12);
  % grid on;
  % set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5);
  % print -dpng 'mlp1h_accuracy_plot.png';
endfunction

function s = softmax(x)
  ex = exp(x - max(x,[],2));
  s = ex ./ sum(ex,2);
endfunction
