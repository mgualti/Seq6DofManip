function PlotResultsMultiple()

    %% Parameters
    
    resultFiles = dir('../results-*.mat');
    episodeBlock = 1000; worstReturn = 0; bestReturn = 4;
    
    saveFilePrefix = '2019-01-23-B';
    saveFilePostfixes = {'AverageReturn', 'ReturnStatistics', 'Loss'};
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
    
    minEpisodeLength = inf;
    nRealizations = length(resultFileNames);
    figs = [figs, figure];  hold('on');
    
    for idx=1:nRealizations
        if length(data{idx}.episodeReturn) < minEpisodeLength
            minEpisodeLength = length(data{idx}.episodeReturn);
        end
        rVis = reshape(data{idx}.episodeReturn, [episodeBlock, ...
            length(data{idx}.episodeReturn)/episodeBlock]);
        rVis = mean(rVis, 1);
        episode = (0:episodeBlock:length(data{idx}.episodeReturn)-1);
        %plot(episode, rVis);
        plot(episode, rVis, '-x', 'linewidth', 2);
    end
    
    %xlim([0, minEpisodeLength]);
    ylim([worstReturn, bestReturn]); grid('on');
    xlabel('Episode'); ylabel('Average Return');
    title(['Return Averaged over 1 Realization and ' num2str(episodeBlock) '-Episode Blocks']);
    
    if nRealizations <= 10
        legend(resultFileNames);
        legend('Location', 'best');
        legend('boxoff');
    end
    
    %% Statistics
    
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
        plot(episode, loss, '-x', 'linewidth', 2);
    end
     
    xlabel('Episode'); ylabel('Loss'); grid('on');
    title('Average Loss Per Training Round');
    
    if nRealizations <= 10
        legend(resultFileNames);
        legend('Location', 'best');
        legend('boxoff');
    end
    
    %% Print Values
    
    for idx=1:nRealizations
        disp(['Average return for ' resultFileNames{idx} ': ' ...
            num2str(mean(data{idx}.episodeReturn(data{idx}. ...
            unbiasOnEpisode:end)))]);
    end
    
    for idx=1:nRealizations
        disp(['Total time for ' resultFileNames{idx} ': ' ...
            num2str(sum(data{idx}.episodeTime) / 3600) ' hours']);
    end
    
    %% Saving Images
    
    for idx=1:length(figsToSave)
        saveas(figs(figsToSave(idx)), ['../../Notebook/figures-4/' ...
            saveFilePrefix '-' saveFilePostfixes{figsToSave(idx)} '.png']);
    end
    