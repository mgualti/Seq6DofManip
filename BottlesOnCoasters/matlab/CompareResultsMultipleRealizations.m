function CompareResultsMultipleRealizations()

    %% Parameters
    
    resultDirs = {'../results-no-clutter','../results-clutter'};
    resultLegend = {'No Distractors', 'Distractors'};
    episodeBlock = 1000; worstReturn = 0; bestReturn = 4;
    
    saveFilePrefix = '2019-01-24';
    saveFilePostfixes = {'AverageReturn'};
    figsToSave = [];

    %% Load

    close('all');
    figs = [];
    
    data = cell(1, length(resultDirs));
    for idx=1:length(resultDirs)
        resultFiles = dir([resultDirs{idx} '/*.mat']);
        data{idx} = cell(1, length(resultFiles));
        for jdx=1:length(resultFiles)
            name = resultFiles(jdx).name(1:end-4);
            fullName = [resultFiles(jdx).folder '/' name];
            data{idx}{jdx} = load(fullName);
        end
    end

    %% Plot Return
    
    figs = [figs, figure];  hold('on');
    plots = zeros(1, length(resultDirs));
    colors = get(gca, 'ColorOrder');
    maxMinRealizationLength = -inf;
    
    for idx=1:length(resultDirs)
        
        nRealizations = length(data{idx});
        
        % determine length of shortest learning curve in this realization
        minRealizationLength = inf;
        for jdx=1:nRealizations
            if length(data{idx}{jdx}.episodeReturn) < minRealizationLength
                minRealizationLength = length(data{idx}{jdx}.episodeReturn);
            end
        end
        
        % determine longest plot in comparison
        if minRealizationLength > maxMinRealizationLength
            maxMinRealizationLength = minRealizationLength;
        end
        
        Return = zeros(nRealizations, minRealizationLength);
    
        for jdx=1:nRealizations
            Return(jdx, 1:minRealizationLength) = ...
                data{idx}{jdx}.episodeReturn(1:minRealizationLength);
        end

        uR = mean(Return, 1);
        sR = std(Return, 0, 1);

        uRvis = reshape(uR, [episodeBlock, length(uR)/episodeBlock]);
        uRvis = mean(uRvis, 1);
        sRvis = reshape(sR, [episodeBlock, length(sR)/episodeBlock]);
        sRvis = mean(sRvis, 1);

        x = (0:episodeBlock:length(uR)-1);

        fill([x'; flipud(x')],[(uRvis-sRvis)'; flipud((uRvis+sRvis)')], ...
            colors(idx, :), 'linestyle','none'); alpha(0.30);
        plots(idx) = plot(x, uRvis, 'color', colors(idx, :),'linewidth', 2);
    end
    
    xlim([0, maxMinRealizationLength-episodeBlock]);
    ylim([worstReturn, bestReturn]); grid('on');
    xlabel('Episode', 'FontWeight', 'bold');
    ylabel('Sum of Rewards', 'FontWeight', 'bold');
    
    if ~isempty(resultLegend)
        legend(plots, resultLegend);
        legend('Location', 'best');
        legend('boxoff');
    end
    
    %% Saving Images
    
    for idx=1:length(figsToSave)
        saveas(figs(figsToSave(idx)), ['../../Notebook/figures-4/' ...
            saveFilePrefix '-' saveFilePostfixes{figsToSave(idx)} '.png']);
    end
    