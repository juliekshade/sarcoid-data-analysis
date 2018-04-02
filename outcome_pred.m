% sarcoidosis - look at pacing sites, clinical data
% import data
clear all;
close all;

[~, labels, ~] = xlsread('Sarcoidosis.xlsx', 'Regression_alldata', 'A2:C68');
data = xlsread('Sarcoidosis.xlsx', 'Regression_alldata', 'D2:V69');

Y = data(2, :); % look at SVT and NSVT first
X = [data(3:15,:); data(48:67,:)]';

% find optimal tree complexity level
rng(1); % For reproducibility
MdlDeep = fitctree(X,Y,'CrossVal','on','MergeLeaves','off',...
    'MinParentSize',1,'Surrogate','on');
MdlStump = fitctree(X,Y,'MaxNumSplits',1,'CrossVal','on', 'Surrogate', 'on');

n = size(X,1);
m = floor(log(n - 1)/log(3));
learnRate = [0.1 0.25 0.5 1];
numLR = numel(learnRate);
maxNumSplits = 3.^(0:m);
numMNS = numel(maxNumSplits);
numTrees = 150;
Mdl = cell(numMNS,numLR);

for k = 1:numLR
    for j = 1:numMNS
        t = templateTree('MaxNumSplits',maxNumSplits(j));
        Mdl{j,k} = fitcensemble(X,Y,'NumLearningCycles',numTrees,...
            'Learners',t,'Leaveout','on','LearnRate',learnRate(k));
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
    h.YLim = [0 1];
    xlabel 'Number of trees';
    ylabel 'Cross-validated misclass. rate';
    title(sprintf('MaxNumSplits = %0.3g', maxNumSplits(mnsPlot(k))));
    hold off;
end
hL = legend([cellstr(num2str(learnRate','Learning Rate = %0.2f'));...
        'Deep Tree';'Stump';'Min. misclass. rate']);
hL.Position(1) = 0.6;

[minErr,minErrIdxLin] = min(error(:));
[idxNumTrees,idxMNS,idxLR] = ind2sub(size(error),minErrIdxLin);

fprintf('\nMin. misclass. rate = %0.5f',minErr)
fprintf('\nOptimal Parameter Values:\nNum. Trees = %d',idxNumTrees);
fprintf('\nMaxNumSplits = %d\nLearning Rate = %0.2f\n',...
    maxNumSplits(idxMNS),learnRate(idxLR))

tFinal = templateTree('MaxNumSplits',maxNumSplits(idxMNS));
MdlFinal = fitcensemble(X,Y,'NumLearningCycles',idxNumTrees,...
    'Learners',tFinal,'LearnRate',learnRate(idxLR))

imp = predictorImportance(MdlFinal);
label = predict(MdlFinal,X);

tp = sum(label(1:11));
fp = sum(label(12:19));
tn = 8 - fp;
fn = 11 - tp;

acc = (tp + tn)/(tp+fp+tn+fn)
sens = tp/(tp+fn)
spec = tn/(tn+fp)

