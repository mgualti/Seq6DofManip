function PlotResultsMultiple()

    %% Parameters
    
    resultFiles = dir('../results-*.mat');
    episodeBlock = 1000; worstReturn = 0; bestReturn = 4;
    
    saveFilePrefix = '2019-02-08';
    saveFilePostfixes = {'AverageReturn', 'NumberOfPlacedObjects', ...
        'ReturnStatistics', 'Loss'};
    figsToSave = [];

    %% Load

    close('all');
    if length(resultFiles) < 2
        disp(['Only ' num2str(length(resultFiles)) ' result files.']);
        return;
    end
    
    data = cell(1, length(resultFiles));
    resultFileNames = cell(1, length(resultFiles));
    for idx=1:length(resultFiles)
        name = resultFiles(idx).name(1:end-4);
        fullName = [resultFiles(idx).folder '/' name];
        resultFileNames{idx} = name;
        data{idx} = load(fullName);
    end
    figs = [];

    %% Plot Return
    
    figs = [figs, PlotMultiple(resultFileNames, data, 'episodeReturn', ...
        episodeBlock, worstReturn, bestReturn, 'Return')];
    figs = [figs, PlotMultiple(resultFileNames, data, ...
        'nGraspedObjects', episodeBlock, 0, 2, 'Number of Grasped Objects')];
    figs = [figs, PlotMultiple(resultFileNames, data, ...
        'nPlacedObjects', episodeBlock, 0, 2, 'Number of Placed Objects')];
    
    %% Statistics
    
    nRealizations = length(resultFileNames);
    minEpisodes = inf;
    for idx=1:nRealizations
        if length(data{idx}.episodeReturn) <  minEpisodes
            minEpisodes = length(data{idx}.episodeReturn);
        end
    end
    
    Return = zeros(nRealizations, minEpisodes);
    
    for idx=1:nRealizations
        Return(idx, 1:minEpisodes) = data{idx}.episodeReturn(1:minEpisodes);
    end
    
    uR = mean(Return, 1);
    sR = std(Return, 0, 1);
    
    uRvis = reshape(uR, [episodeBlock, length(uR)/episodeBlock]);
    uRvis = mean(uRvis, 1);
    sRvis = reshape(sR, [episodeBlock, length(sR)/episodeBlock]);
    sRvis = mean(sRvis, 1);
    
    x = (1:episodeBlock:length(uR));
    
    figs = [figs, figure]; hold('on');
    fill([x';flipud(x')],[(uRvis-sRvis)';flipud((uRvis+sRvis)')], ...
        [0.2, 0.2, 0.9], 'linestyle','none'); alpha(0.50);
    plot(x, uRvis, 'color',[0.2,0.2,0.9],'linewidth', 2);
    
    xlim([x(1), x(end)]); ylim([0, bestReturn]); grid('on');
    xlabel('Episode'); ylabel('Average Return');
    title(['Return Averaged over ' num2str(nRealizations) ' Realizations']);
    
    %% Loss
    
    figs = [figs, figure]; hold('on');
    
    maxEpisodes = -inf;
    for idx=1:nRealizations
        if length(data{idx}.episodeReturn) > maxEpisodes
            maxEpisodes = length(data{idx}.episodeReturn);
        end
        episode = data{idx}.trainEvery:data{idx}.trainEvery:length( ...
            data{idx}.episodeReturn);
        loss = sum(data{idx}.losses, 2);
        plot(episode, loss, '-', 'linewidth', 2);
    end
    
    xlim([episode(1), maxEpisodes]);
    xlabel('Episode'); ylabel('Loss'); grid('on');
    title('Average Loss Per Training Round');
    
    if nRealizations <= 10
        legend(resultFileNames);
        legend('Location', 'best');
        legend('boxoff');
    end
    
    %% Time
    
    disp('Time ---------------------------------------------------------');
    for idx=1:nRealizations
        disp([resultFileNames{idx} ': ' ...
            num2str(sum(data{idx}.episodeTime) / 3600) 'h.']);
    end
    
    %% Saving Images
    
    for idx=1:length(figsToSave)
        saveas(figs(figsToSave(idx)), ['../../Notebook/figures-4/' ...
            saveFilePrefix '-' saveFilePostfixes{figsToSave(idx)} '.png']);
    end
    
end
    
function [fig, maxEpisodes] = PlotMultiple(fileNames, data, metricName, ...
    episodeBlock, worstReturn, bestReturn, titleName)

    fig = figure;  hold('on');
    nRealizations = length(fileNames);
    maxEpisodes = -Inf;
    disp([titleName '--------------------------------------------------']);
    
    for idx=1:nRealizations
        metric = eval(['data{idx}.', metricName]);
        if length(metric) > maxEpisodes, maxEpisodes = length(metric); end
        rVis = reshape(metric, [episodeBlock, length(metric)/episodeBlock]);
        rVis = mean(rVis, 1);
        episode = (episodeBlock:episodeBlock:length(metric));
        plot(episode, rVis, '-', 'linewidth', 2);
        unbiasedMetric = mean(metric(data{idx}.unbiasOnEpisode+1:end));
        disp([fileNames{idx} ': ' num2str(unbiasedMetric) '.']);
    end
    
    xlim([episodeBlock, maxEpisodes]); ylim([worstReturn, bestReturn]);
    xlabel('Episode'); ylabel(titleName); grid('on');
    title([titleName ' Averaged over ' num2str(episodeBlock) ...
        '-Episode Blocks']);
    
    if nRealizations <= 10
        legend(fileNames);
        legend('Location', 'best');
        legend('boxoff');
    end
end