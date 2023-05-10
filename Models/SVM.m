function [accuracy, precision, recall, f1] = himate()

% Read in dataset
data = readtable('EcoPreprocessed.csv');

% Preprocess review text
docs = preprocessText(data.review);

% Convert division values to numerical labels
labels = zeros(size(data,1),1);
labels(strcmp(data.division,'positive')) = 1;
labels(strcmp(data.division,'neutral')) = 2;
labels(strcmp(data.division,'negative')) = 3;

% Split data into training and testing sets
cv = cvpartition(size(data,1),'HoldOut',0.2);
trainIDx = cv.training;
testIDx = cv.test;
trainDocs = docs(trainIDx);
trainLabels = labels(trainIDx);
testDocs = docs(testIDx);
testLabels = labels(testIDx);

% Create bag-of-words model
bags = bagOfWords(trainDocs);

% Remove infrequent words
bags = removeInfrequentWords(bags,5);

% Convert to matrix
XTrain = encode(bags,trainDocs);
XTest = encode(bags,testDocs);

% Train SVM model
svmModel = fitcecoc(XTrain,trainLabels);

% Test SVM model
predictions = predict(svmModel,XTest);

% Calculate evaluation metrics
accuracy = sum(predictions == testLabels) / numel(testLabels);

cm = confusionmat(testLabels, predictions);
precision = diag(cm)./sum(cm,1)';
recall = diag(cm)./sum(cm,2);
f1 = 2.*(precision.*recall)./(precision+recall);

% Plot evaluation metrics
figure
bar([accuracy, mean(precision), mean(recall), mean(f1)])
xticklabels({'Accuracy', 'Precision', 'Recall', 'F1 Score'})
ylabel('Metric score')
title('Evaluation Metrics')

end

function docs = preprocessText(textData)

% Tokenize text
docs = tokenizedDocument(textData);

% Remove stop words
docs = removeStopWords(docs);

% Stem words
docs = normalizeWords(docs,'Style','stem');

% Remove words with 2 or fewer characters
docs = removeShortWords(docs,2);

% Remove words with 15 or more characters
docs = removeLongWords(docs,15);

end


