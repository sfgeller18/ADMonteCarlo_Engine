    #include <unistd.h>
    #include "StochasticProcess.h"
    #include "mpfr_helpers.hpp"

    #define precision 12
    
    static void printEvolution(const StochasticProcess& process, const mpfr_t timeStep, int numSteps, std::string file_name) {
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
            mpfr_t jump;
            mpfr_init2(jump, precision); // Initialize with a specified precision
            mpfr_set(jump, *mapped.getRandomStep(timeStep), MPFR_RNDN);
            mapped.step(jump);
            mpfr_t temp;
            mpfr_init2(temp, precision);
            mpfr_set_d(temp, i, MPFR_RNDN);
            mpfr_mul(temp, temp, timeStep, MPFR_RNDN);
            // Print values to the file
            outputFile << mpfrToString(temp) << "," << mpfrToString(*mapped.getPosition()) << ","<< mpfrToString(*mapped.getDrift()) << "," << mpfrToString(*mapped.getVariance()) << "\n";
            mpfr_clear(jump);
            mpfr_clear(temp);
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

static void printHestonEvolution(const HestonProcess& process, const mpfr_t timeStep, int numSteps, std::string file_name) {
        const char* outputDir = "../output";
        std::string file = std::string(outputDir).append("/"+file_name);
        const char* filePath = file.c_str();
        HestonProcess mapped(process);
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
        for (int i = 1; i < numSteps; ++i) {
            mapped.volStep(timeStep);
            mapped.HestonStep(timeStep);
            mpfr_t temp;
            mpfr_init2(temp, precision);
            mpfr_set_si(temp, i, MPFR_RNDN);
            mpfr_mul(temp, temp, timeStep, MPFR_RNDN);
            // Print values to the file
            outputFile << mpfrToString(temp) << "," << mpfrToString(*mapped.getPosition()) << ","
                       << mpfrToString(*(mapped.getVolatility()).getPosition()) << "\n";
            mpfr_clear(temp);
        }

        outputFile.close();

         const char* gnuplotScriptPath = "/home/sfgeller18/projects/ADMonteCarlo_Engine/src/plot_script.gp"; // Replace with actual path
        
        std::ofstream clearScriptFile(gnuplotScriptPath, std::ios::trunc); // Open the script in write mode to clear its contents
        clearScriptFile.close();
        
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

    }

