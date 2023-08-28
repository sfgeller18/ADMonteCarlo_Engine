    #include <unistd.h>
    #include "StochasticProcess.h"
    
    static void printEvolution(const StochasticProcess& process, double timeStep, int numSteps, std::string file_name) {
        const char* outputDir = "../output";
        std::string file = std::string(outputDir).append("/"+file_name);
        const char* filePath = file.c_str();
        StochasticProcess mapped(process);
    // Check if the program has write permission for the output directory
        if (access(outputDir, W_OK) != 0) {
                std::cerr << "Error: No write permission for " << outputDir << std::endl;
            exit(1);
        }

        // Create and open the output file
        std::ofstream outputFile(filePath);
        if (!outputFile.is_open()) {
            std::cerr << "Failed to open output file." << std::endl;
            exit(1);
        }

        if (access(filePath, W_OK) != 0) {
            std::cerr << "Error: No write permission for " << filePath << std::endl;
            exit(1);
        }

        // Print headers for the values
        outputFile << "Time Step,Position,Drift,Variance\n";

        // Print the evolution over time steps
        for (int i = 0; i < numSteps; ++i) {
            double jump = mapped.getRandomStep(timeStep);
            mapped.step(jump);
            // Print values to the file
            outputFile << i * timeStep << "," << mapped.getPosition() << ","
                       << mapped.getDrift() << "," << mapped.getVariance() << "\n";
        }

        outputFile.close();

         const char* gnuplotScriptPath = "/home/sfgeller18/projects/ADMonteCarlo_Engine/src/plot_script.gp"; // Replace with actual path

        // Modify the Gnuplot script to plot the data
        std::ofstream plotScriptFile(gnuplotScriptPath, std::ios::app); // Open the script in append mode
        if (plotScriptFile.is_open()) {
            // Add commands to plot the data
             plotScriptFile << "set term png\n";
            plotScriptFile << "set output '../output/simulation_plot.png'\n";
            plotScriptFile << "set datafile separator \",\"\n";
            plotScriptFile << "plot \"" << filePath << "\" every::1 using 1:2 with lines title 'Position'\n";
            // Add more plot commands as needed
            plotScriptFile.close();
        } else {
            std::cerr << "Failed to open Gnuplot script file." << std::endl;
            exit(1);
        }
        
        // Run the Gnuplot script
        std::string command = "gnuplot ";
        command += gnuplotScriptPath;
        int returnValue = std::system(command.c_str());

        if (returnValue != 0) {
            std::cerr << "Failed to execute the Gnuplot script." << std::endl;
                exit(1);
        }
        std::ofstream clearScriptFile(gnuplotScriptPath, std::ios::trunc); // Open the script in write mode to clear its contents
        clearScriptFile.close();
    }