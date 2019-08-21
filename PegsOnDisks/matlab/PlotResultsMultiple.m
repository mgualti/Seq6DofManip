function PlotResultsMultiple()

    %% Parameters
    
    resultFiles = dir('../results-shaped-reward/results-*.mat');
    episodeBlock = 1000; worstReturn = 0; bestReturn = 4;
    
    saveFilePrefix = '2019-02-01-B';
    saveFilePostfixes = {'AverageReturn', 'GraspReturn', 'PlaceReturn', ...
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
    
    figs = [figs PlotMultiple(resultFileNames, data, 'episodeReturn', ...
        episodeBlock, worstReturn, bestReturn, 'Return')];
    figs = [figs PlotMultiple(resultFileNames, data, 'nPlacedObjects', ...
        episodeBlock, 0, 2, 'Number of Placed Objects')];
    
    %% Statistics
    
    nRealizations = length(resultFileNames);
    minEpisodeLength = inf;
    for idx=1:nRealizations
        if length(data{idx}.episodeReturn) <  minEpisodeLength
            minEpisodeLength = length(data{idx}.episodeReturn);
        end
    end
    
    Return = zeros(nRealizations, minEpisodeLength);
    
    for idx=1:nRealizations
        Return(idx, 1:minEpisodeLength) = data{idx}.episodeReturn(1:minEpisodeLength);
    end
    
    uR = mean(Return, 1);
    sR = std(Return, 0, 1);
    
    uRvis = reshape(uR, [episodeBlock, length(uR)/episodeBlock]);
    uRvis = mean(uRvis, 1);
    sRvis = reshape(sR, [episodeBlock, length(sR)/episodeBlock]);
    sRvis = mean(sRvis, 1);
    
    x = (0:episodeBlock:length(uR)-1);
    
    figs = [figs, figure]; hold('on');
    fill([x';flipud(x')],[(uRvis-sRvis)';flipud((uRvis+sRvis)')], ...
        [0.2, 0.2, 0.9], 'linestyle','none'); alpha(0.50);
    plot(x, uRvis, 'color',[0.2,0.2,0.9],'linewidth', 2);
    
    ylim([0, bestReturn]); grid('on');
    xlabel('Episode'); ylabel('Average Return');
    title(['Return Averaged over ' num2str(nRealizations) ' Realizations']);
    
    %% Loss
    
    figs = [figs, figure]; hold('on');
    
    for idx=1:nRealizations
        episode = (0:data{idx}.trainEvery:length(data{idx}.episodeReturn)-1);
        loss = sum(data{idx}.losses, 2);
        plot(episode, loss, '-', 'linewidth', 2);
    end
     
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
    
function fig = PlotMultiple(fileNames, data, metricName, episodeBlock, ...
    worstReturn, bestReturn, titleName)

    fig = figure;  hold('on');
    nRealizations = length(fileNames);
    maxEpisodes = -Inf;
    disp([titleName '--------------------------------------------------']);
    
    for idx=1:nRealizations
        metric = eval(['data{idx}.', metricName]);
        if length(metric) > maxEpisodes, maxEpisodes = length(metric); end
        rVis = reshape(metric, [episodeBlock, length(metric)/episodeBlock]);
        rVis = mean(rVis, 1);
        episode = (0:episodeBlock:length(metric)-1);
        plot(episode, rVis, '-', 'linewidth', 2);
        unbiasedMetric = mean(metric(data{idx}.unbiasOnEpisode+1:end));
        disp([fileNames{idx} ': ' num2str(unbiasedMetric) '.']);
    end
    
    xlim([0, maxEpisodes-episodeBlock]); ylim([worstReturn, bestReturn]);
    xlabel('Episode'); ylabel(titleName); grid('on');
    title([titleName ' Averaged over ' num2str(episodeBlock) ...
        '-Episode Blocks']);
    
    if nRealizations <= 10
        legend(fileNames);
        legend('Location', 'best');
        legend('boxoff');
    end
end