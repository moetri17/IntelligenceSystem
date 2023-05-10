% Load data
data = readtable('EcoPreprocessed.csv');
data = data(:, {'review', 'division'});
data = rmmissing(data);
numRows = size(data, 1);
sentiments = strings(numRows, 1);
for i = 1:numRows
    division = data.division(i);
    if strcmp(division, 'positive')
        sentiments(i) = "Positive";
    elseif strcmp(division, 'negative')
        sentiments(i) = "Negative";
    else
        sentiments(i) = "Neutral";
    end
end
data.Sentiment = categorical(sentiments);

% Tokenize data
num_words = 5000;
tokenizer = tokenizedDocument(data.review);
enc = wordEncoding(tokenizer);
X = doc2sequence(enc, tokenizer, 'Length', num_words);
max_length = max(cellfun(@(x) numel(x), X));
X = cellfun(@(x) [x, zeros(1, max_length - numel(x))], X, 'UniformOutput', false);
X = reshape(X, [1, numRows]);

% Create target variable
y = categorical(data.Sentiment);

% Split into training and testing sets
cv = cvpartition(numRows, 'HoldOut', 0.15);
idxTrain = cv.training;
idxTest = cv.test;
X_train = X(idxTrain);
X_test = X(idxTest);
y_train = y(idxTrain);
y_test = y(idxTest);

% Define CNN network layers
inputSize = [1 num_words 1];
numFilters = 32;
filterSize = [3 3];
poolSize = [2 2];
numClasses = 2;

layers = [    imageInputLayer(inputSize)    convolution2dLayer(filterSize, numFilters, 'Padding', 'same')    reluLayer    maxPooling2dLayer(poolSize, 'Stride', 2)    convolution2dLayer(filterSize, numFilters*2, 'Padding', 'same')    reluLayer    maxPooling2dLayer(poolSize, 'Stride', 2)    convolution2dLayer(filterSize, numFilters*4, 'Padding', 'same')    reluLayer    maxPooling2dLayer(poolSize, 'Stride', 2)    fullyConnectedLayer(numClasses)    softmaxLayer    classificationLayer];

% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs', 4, ...
    'MiniBatchSize', 64, ...
    'ValidationData', {X_test, y_test}, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

% Train the CNN network
net = trainNetwork(X_train, y_train, layers, options);

% Evaluate the network
y_pred = classify(net, X_test);
accuracy = sum(y_pred == y_test) / numel(y_test);
disp(['Accuracy: ', num2str(accuracy)]);
