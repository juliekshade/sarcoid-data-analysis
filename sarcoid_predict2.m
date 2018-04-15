% sarcoidosis - try to predict if reentry will occur at given pacing site 
% based on 
% import data
clear all;
close all;

[~, labels, ~] = xlsread('Sarcoidosis.xlsx', 'AHA scarinf', 'G2:H495');
data = xlsread('Sarcoidosis.xlsx', 'AHA scarinf', 'I2:T495');

Y = data(:, 11); % look at SVT and NSVT first
X = [data(:,1:9)];
rng(1); % For reproducibility

% split test training data
idx = randperm(494);
indexToGroup1 = (idx<=100);
indexToGroup2 = (idx>100);
Xtest= X(indexToGroup1,:);
X = X(indexToGroup2,:);
Ytest= Y(indexToGroup1,:);
Y = Y(indexToGroup2,:);

% find optimal tree complexity level
MdlDeep = fitctree(X,Y,'CrossVal','on','MergeLeaves','off',...
    'MinParentSize',1,'Surrogate','on');
MdlStump = fitctree(X,Y,'MaxNumSplits',1,'CrossVal','on', 'Surrogate', 'on');

n = size(X,2);
m = floor(log(n - 1)/log(2));
learnRate = [0.1 0.25 0.5 1];
numLR = numel(learnRate);
maxNumSplits = 2.^(0:m);
numMNS = numel(maxNumSplits);
numTrees = 9;
Mdl = cell(numMNS,numLR);

for k = 1:numLR
    for j = 1:numMNS
        t = templateTree('MaxNumSplits',maxNumSplits(j));
        Mdl{j,k} = fitcensemble(X,Y,'NumLearningCycles',numTrees,...
            'Learners',t,'kFold',5,'LearnRate',learnRate(k));
    end
end

kflAll = @(x)kfoldLoss(x,'Mode','cumulative');
errorCell = cellfun(kflAll,Mdl,'Uniform',false);
error = reshape(cell2mat(errorCell),[numTrees numel(maxNumSplits) numel(learnRate)]);
errorDeep = kfoldLoss(MdlDeep)
errorStump = kfoldLoss(MdlStump)

mnsPlot = [1 round(numel(maxNumSplits)/2) numel(maxNumSplits)];
figure;
for k = 1:3
    subplot(2,2,k);
    plot(squeeze(error(:,mnsPlot(k),:)),'LineWidth',2);
    axis tight;
    hold on;
    h = gca;
    plot(h.XLim,[errorDeep errorDeep],'-.b','LineWidth',2);
    plot(h.XLim,[errorStump errorStump],'-.r','LineWidth',2);
    plot(h.XLim,min(min(error(:,mnsPlot(k),:))).*[1 1],'--k');
    h.YLim = [0 .2];
    xlabel 'Number of trees';
    ylabel 'Cross-validated misclass. rate';
    title(sprintf('MaxNumSplits = %0.3g', maxNumSplits(mnsPlot(k))));
    hold off;
end
hL = legend([cellstr(num2str(learnRate','Learning Rate = %0.2f'));...
        'Deep Tree';'Stump';'Min. misclass. rate']);
hL.Position(1) = 0.6;

[minErr,minErrIdxLin] = min(error(:));
find(error(:)==minErr)
[idxNumTrees,idxMNS,idxLR] = ind2sub(size(error),minErrIdxLin);

fprintf('\nMin. misclass. rate = %0.5f',minErr)
fprintf('\nOptimal Parameter Values:\nNum. Trees = %d',idxNumTrees);
fprintf('\nMaxNumSplits = %d\nLearning Rate = %0.2f\n',...
    maxNumSplits(idxMNS),learnRate(idxLR))

tFinal = templateTree('MaxNumSplits',maxNumSplits(idxMNS));
MdlFinal = fitcensemble(X,Y,'NumLearningCycles',idxNumTrees,...
    'Learners',tFinal,'LearnRate',learnRate(idxLR))

view(MdlFinal.Trained{1,1}.CompactRegressionLearner,'Mode','graph')

imp = predictorImportance(MdlFinal)

label = predict(MdlFinal,Xtest)

e = classperf(Ytest,label)

D_hat = find(imp~=0)

Xreg = [ones(size(X,1),1), X(:,D_hat)];
w = pinv(Xreg'*Xreg)*Xreg'*Y;
Xtest_cond = [ones(size(Xtest,1),1), Xtest(:,D_hat)];
p_hat_test = 1./(1+exp(w'*Xtest_cond')); % calculate discriminant
for t = 0:.00001:1
    Yhat_test = zeros(1,size(Xtest,1));
    Yhat_test(find(p_hat_test <= t))=1;
    CP = classperf(Ytest, Yhat_test);
    sens(int16(t*100000)+1) = CP.Sensitivity;
    spec(int16(t*100000)+1) = CP.Specificity;
    acc(int16(t*100000)+1) = CP.CorrectRate;
end

t = 0:.00001:1;

% plot ROC curve
figure(2)
plot(1-spec, sens)
hold on;
xlabel('1-spec(t)')
ylabel('sens(t)')
title('ROC Curve for NodalStatus Classifier')

figure(3)
plot(t,acc)
hold on;
xlabel('1-spec(t)')
ylabel('sens(t)')
title('ROC Curve for NodalStatus Classifier')

% try entropy of scar as parameter! sum(plnp)
% use pacing site calc to determine output (y= clinical vt)
% deep network?
% include pacing site loc?