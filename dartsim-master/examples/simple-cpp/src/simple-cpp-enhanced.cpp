/*******************************************************************************
 * DARTSim Mission Simulator - Enhanced Version
 *
 * This version outputs detailed state information for visualization
 ******************************************************************************/
#include <dartsim/Simulator.h>
#include <iostream>
#include <getopt.h>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <sstream>

using namespace std;
using namespace dart::sim;

using myclock = chrono::high_resolution_clock;

enum ARGS {
	LOOKAHEAD_horizon
};

static struct option long_options[] = {
    {"lookahead-horizon",  required_argument, 0,  LOOKAHEAD_horizon },
    {0, 0, 0, 0 }
};

static void usage() {
	cout << "options: " << endl;
	cout << "\t[simulator options] [-- [adaptation manager options]]" << endl;
	Simulator::usage();
	cout << "valid adaptation manager options are:" << endl;
	int opt = 0;
	while (long_options[opt].name != 0) {
		cout << "\t--" << long_options[opt].name;
		if (long_options[opt].has_arg == required_argument) {
			cout << "=value";
		}
		cout << endl;
		opt++;
	}
	exit(EXIT_FAILURE);
}

// Helper function to output state in JSON-like format
void outputState(const TeamState& state, const vector<bool>& threats, const vector<bool>& targets) {
	cout << "STATE:" << endl;
	cout << "  position: " << state.position.x << ";" << state.position.y << endl;
	cout << "  direction: " << state.directionX << ";" << state.directionY << endl;
	cout << "  altitude: " << state.config.altitudeLevel << endl;
	cout << "  formation: " << (state.config.formation == TeamConfiguration::TIGHT ? "TIGHT" : "LOOSE") << endl;
	cout << "  ecm: " << (state.config.ecm ? "true" : "false") << endl;
	cout << "  ttcIncAlt: " << state.config.ttcIncAlt << endl;
	cout << "  ttcDecAlt: " << state.config.ttcDecAlt << endl;
	
	// Output threats
	cout << "  threats: [";
	for (size_t i = 0; i < threats.size(); ++i) {
		if (i > 0) cout << ",";
		cout << (threats[i] ? "1" : "0");
	}
	cout << "]" << endl;
	
	// Output targets
	cout << "  targets: [";
	for (size_t i = 0; i < targets.size(); ++i) {
		if (i > 0) cout << ",";
		cout << (targets[i] ? "1" : "0");
	}
	cout << "]" << endl;
}

int main(int argc, char** argv) {
	int horizon = 5;

	// instantiate sim first

	/*
	 * Split all command-line options first
	 * All the options before a -- arg are for the sim, the rest are for
	 * the adaptation manager
	 */
	int simArgc = 0;

	while (simArgc < argc) {
		if (strcmp(argv[simArgc++], "--") == 0) {
			simArgc--;
			break;
		}
	}

	int amArgc = argc - simArgc;
	if (amArgc) {
		argv[simArgc] = argv[0];
		char **amArgv = argv + simArgc;


		while (1) {
			int option_index = 0;

			auto c = getopt_long(amArgc, amArgv, "", long_options, &option_index);

			if (c == -1) {
				break;
			}

			switch (c) {
			case LOOKAHEAD_horizon:
				horizon = atoi(optarg);
				if (horizon < 1) {
					cout << "error: horizon must be >= 1" << endl;
					usage();
				}
				break;
			default:
				usage();
			}
		}

		if (optind < amArgc) {
			usage();
		}
	}

	optind = 1; // reset getopt scanning
	argv[simArgc] = nullptr;

	Simulator *dartsim = Simulator::createInstance(simArgc, argv);
	if (!dartsim) {
		usage();
	}

	auto simParams = dartsim->getParameters();
	const unsigned minAltitude = 1;
	const unsigned maxAltitude = simParams.altitudeLevels;

	while (!dartsim->finished()) {
		auto startTime = myclock::now();
		auto state = dartsim->getState();
		auto threats = dartsim->readForwardThreatSensor(horizon);
		auto targets = dartsim->readForwardTargetSensor(horizon);
		
		// Output detailed state information
		outputState(state, threats, targets);
		
		cout << "current position: " << state.position << endl;

		Simulator::TacticList tactics;
		bool threatAhead = any_of(threats.begin(), threats.end(), [](bool p){return p;});
		if (threatAhead && state.config.altitudeLevel < maxAltitude) {
			tactics.insert(Simulator::INC_ALTITUDE);
		} else {
			bool targetAhead = any_of(targets.begin(), targets.end(), [](bool p){return p;});
			if (targetAhead && state.config.altitudeLevel > minAltitude) {
				tactics.insert(Simulator::DEC_ALTITUDE);
			}
		}

		if (threats[0]) { // is there an immediate threat?
			if (state.config.formation != TeamConfiguration::Formation::TIGHT) {
				tactics.insert(Simulator::GO_TIGHT);
			}
		} else if (state.config.formation != TeamConfiguration::Formation::LOOSE) {
			tactics.insert(Simulator::GO_LOOSE);
		}

		// Output tactics
		for (const auto& tactic : tactics) {
			cout << "executing tactic " << tactic << endl;
		}

		auto delta = myclock::now() - startTime;
		double deltaMsec = chrono::duration_cast<chrono::duration<double, std::milli>>(delta).count();

		dartsim->step(tactics, deltaMsec);
	}

	auto results = dartsim->getResults();
	if (!results.destroyed) {
		cout << "Total targets detected: " << results.targetsDetected << endl;
	}

	cout << dartsim->getScreenOutput();

	const std::string RESULTS_PREFIX = "out:";
	cout << RESULTS_PREFIX << "destroyed=" << results.destroyed << endl;
	cout << RESULTS_PREFIX << "targetsDetected=" << results.targetsDetected << endl;
	cout << RESULTS_PREFIX << "missionSuccess=" << results.missionSuccess << endl;

	cout << "csv," << results.targetsDetected << ',' << results.destroyed
			<< ',' << results.whereDestroyed.x
			<< ',' << results.missionSuccess
			<< ',' << results.decisionTimeAvg
			<< ',' << results.decisionTimeVar
			<<  endl;

	delete dartsim;

	return 0;
}

