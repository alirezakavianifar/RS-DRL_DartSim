/*******************************************************************************
 * DARTSim Mission Simulator
 *
 * Copyright 2019 Carnegie Mellon University. All Rights Reserved.
 * NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING
 * INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON
 * UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS
 * TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE
 * OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE
 * MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND
 * WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
 * 
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 * 
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution. Please see Copyright notice for non-US Government
 * use and distribution.
 * 
 * Carnegie Mellon® is registered in the U.S. Patent and Trademark Office by
 * Carnegie Mellon University.
 * 
 * This Software includes and/or makes use of Third-Party Software, each subject
 * to its own license. See license.txt.
 * 
 * DM19-0045
 ******************************************************************************/

#include "AdaptInterface.h"
#include "assert.h"
#include <boost/tokenizer.hpp>
#include <set>
#include <map>

#define DEBUG_ADAPT_INTERFACE 0

using namespace json11;
using namespace boost::asio;
using namespace boost::asio::ip;

namespace dart {
namespace sim {

static const std::string UNKNOWN_COMMAND = "error: unknown command\n";
static const std::string INVALID_ARGUMENTS = "error: invalid arguments count\n";
static const std::string COMMAND_SUCCESS = "OK\n";

AdaptInterface::AdaptInterface(dart::sim::Simulator* simulatorP, unsigned port)
		: mSimulatorP(simulatorP),
		  mPort(port),
		  mIOServiceP(nullptr),
		  mEndPointP(nullptr),
		  mAcceptorP(nullptr),
		  mSocketP(nullptr) {
	assert(mSimulatorP != nullptr);
	mIOServiceP = new io_service();
	mEndPointP = new tcp::endpoint(tcp::v4(), mPort);
	mAcceptorP = new tcp::acceptor(*mIOServiceP, *mEndPointP);

	mCommandHandlers["finished"] = std::bind(&AdaptInterface::cmdFinished, this, std::placeholders::_1);
	mCommandHandlers["getState"] = std::bind(&AdaptInterface::cmdGetState, this, std::placeholders::_1);
	mCommandHandlers["readForwardThreatSensor"] = std::bind(&AdaptInterface::cmdReadForwardThreatSensor, this, std::placeholders::_1);
	mCommandHandlers["readForwardTargetSensor"] = std::bind(&AdaptInterface::cmdReadForwardTargetSensor, this, std::placeholders::_1);
	mCommandHandlers["readForwardThreatSensorForObservations"] = std::bind(&AdaptInterface::cmdReadForwardThreatSensorForObservations, this, std::placeholders::_1);
	mCommandHandlers["readForwardTargetSensorForObservations"] = std::bind(&AdaptInterface::cmdReadForwardTargetSensorForObservations, this, std::placeholders::_1);
	mCommandHandlers["step"] = std::bind(&AdaptInterface::cmdStep, this, std::placeholders::_1);
	mCommandHandlers["getResults"] = std::bind(&AdaptInterface::cmdGetResults, this, std::placeholders::_1);
	mCommandHandlers["getScreenOutput"] = std::bind(&AdaptInterface::cmdGetScreenOutput, this, std::placeholders::_1);
	mCommandHandlers["getParameters"] = std::bind(&AdaptInterface::cmdGetParameters, this, std::placeholders::_1);
}

AdaptInterface::~AdaptInterface() {
	if (mSocketP != nullptr) {
		if (mSocketP->is_open()) {
			mSocketP->close();
		}
		delete mSocketP;
	}

	if (mAcceptorP != nullptr) {
		delete mAcceptorP;
	}

	if (mIOServiceP != nullptr) {
		delete mIOServiceP;
	}

	if (mEndPointP != nullptr) {
		delete mEndPointP;
	}
}

void AdaptInterface::serviceClient() {
	mSocketP = new tcp::socket(*mIOServiceP);
	boost::system::error_code errorCode;

	mAcceptorP->accept(*mSocketP, errorCode);

	if (errorCode) {
		throw boost::system::system_error(errorCode);
	}

	std::cout << "Simulation client connected" << std::endl;

	// Set socket options for better reliability
	mSocketP->set_option(boost::asio::ip::tcp::no_delay(true));
	mSocketP->set_option(boost::asio::socket_base::keep_alive(true));
	
	// Ensure socket is in blocking mode and ready
	// The socket should already be blocking by default, but ensure it
	mSocketP->non_blocking(false);

	while (true) {
		try {
			auto cmd = readCmd();
			if (!cmd) {
				// connection closed by client (EOF) - this is normal
				break;
			}
			// Process the command
			handleClientCmd(*cmd);
		} catch (const boost::system::system_error& e) {
			// Handle network errors
			// Check if it's a read error (might be recoverable) vs other errors
			if (e.code() == boost::asio::error::eof || 
			    e.code() == boost::asio::error::connection_reset ||
			    e.code() == boost::asio::error::broken_pipe) {
				// Client disconnected - normal exit
				std::cout << "Client disconnected: " << e.what() << std::endl;
				break;
			} else {
				// Other network errors - log and close connection
				std::cerr << "Network error in serviceClient: " << e.what() << " (code: " << e.code().value() << ")" << std::endl;
				break;
			}
		} catch (const std::exception& e) {
			// Handle other errors - try to send error response and continue
			std::cerr << "Error processing command: " << e.what() << std::endl;
			try {
				// Try to send error response to client
				if (mSocketP && mSocketP->is_open()) {
					sendBytes("error: " + std::string(e.what()) + "\n");
					// Continue processing next command
					continue;
				} else {
					// Socket is closed, exit loop
					break;
				}
			} catch (...) {
				// If we can't send error, connection is broken
				std::cerr << "Failed to send error response, closing connection" << std::endl;
				break;
			}
		}
	}

	if (mSocketP && mSocketP->is_open()) {
		mSocketP->close();
	}
}

std::shared_ptr<std::string> AdaptInterface::readCmd() const {
	std::shared_ptr<std::string> cmd; // default to a nullptr for connection closed
	boost::asio::streambuf buf;
	boost::system::error_code error;
	
	// Ensure socket is open before reading
	if (!mSocketP || !mSocketP->is_open()) {
		std::cerr << "Socket is not open in readCmd" << std::endl;
		return cmd; // Return nullptr - connection is closed
	}
	
	// Read command until newline
	// This will block until data arrives or connection closes
	// Use error_code version to avoid throwing on EOF
	try {
		boost::asio::read_until(*mSocketP, buf, "\n", error);
	} catch (const boost::system::system_error& e) {
		// If read_until throws, capture the error
		error = e.code();
	}
	
	if (!error) {
		// Successfully read command
		cmd.reset(new std::string(boost::asio::buffer_cast<const char*>(buf.data())));

		// remove trailing returns
		cmd->erase(cmd->find_last_not_of("\r\n") + 1);
#if DEBUG_ADAPT_INTERFACE
		std::cout << "Command = [" << *cmd << "] length=" << cmd->length() << std::endl;
#endif
	} else if (error == boost::asio::error::eof) {
		// Client closed connection - this is normal, return nullptr
		std::cout << "Client closed connection" << std::endl;
		// cmd is already nullptr, which is correct
	} else if (error == boost::asio::error::operation_aborted) {
		// Operation was aborted - treat as connection closed
		std::cout << "Read operation aborted" << std::endl;
		// cmd is already nullptr
	} else {
		// Other errors - throw exception so serviceClient can handle it
		// This allows the service loop to catch and potentially recover
		throw boost::system::system_error(error);
	}

	return cmd;
}

void AdaptInterface::sendBytes(const std::string& bytes) const {
#if DEBUG_ADAPT_INTERFACE
	std::cout << "Command Reply is [ " << bytes << " ]" << std::endl;
#endif
	try {
		boost::system::error_code errorCode;
		
		// Ensure socket is still open
		if (!mSocketP || !mSocketP->is_open()) {
			throw boost::system::system_error(boost::asio::error::not_connected);
		}
		
		boost::asio::write(*mSocketP, boost::asio::buffer(bytes + "\n"), errorCode);

		if (errorCode) {
			std::cerr << "Send error: " << errorCode.message() << std::endl;
			throw boost::system::system_error(errorCode);
		}
		
		// boost::asio::write already sends data immediately, no flush needed for TCP sockets
	} catch (const boost::system::system_error& e) {
		// Re-throw to let caller handle
		throw;
	} catch (const std::exception& e) {
		throw boost::system::system_error(boost::asio::error::fault);
	}
}

void AdaptInterface::handleClientCmd(const std::string& cmd) {
	try {
		typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
		tokenizer tokens(cmd, boost::char_separator<char>(" \n[],"));
		tokenizer::iterator it = tokens.begin();

		if (it != tokens.end()) {
			std::string command = *it;
			std::vector<std::string> args;

			while (++it != tokens.end()) {
#if DEBUG_ADAPT_INTERFACE
				std::cout << "argument " << *it << std::endl;
#endif
				args.push_back(*it);
			}

			auto handler = mCommandHandlers.find(command);
			if (handler == mCommandHandlers.end()) {
				sendBytes(UNKNOWN_COMMAND);
			} else {
				// Wrap command handler in try-catch to ensure we always send a response
				std::string reply;
				try {
					reply = mCommandHandlers[command](args);
					// Ensure we always send something, even if handler returns empty
					if (reply.empty()) {
						reply = "{}";  // Empty JSON object as fallback
					}
					sendBytes(reply);
				} catch (const std::exception& e) {
					// If command handler throws, send error response
					std::string errorMsg = "error: command handler failed: " + std::string(e.what());
					sendBytes(errorMsg);
					std::cerr << "Error in command handler for '" << command << "': " << e.what() << std::endl;
				}
			}
		} else {
			// Empty command - send error
			sendBytes("error: empty command\n");
		}
	} catch (const boost::system::system_error& e) {
		// Network error - rethrow to let serviceClient handle it
		throw;
	} catch (const std::exception& e) {
		// Other errors - try to send error response
		try {
			sendBytes("error: " + std::string(e.what()) + "\n");
		} catch (...) {
			// If we can't send, connection is broken - rethrow
			throw;
		}
	}
}

std::string AdaptInterface::cmdFinished(const std::vector<std::string>& args) {
	std::string result = "";

	if (args.empty()) {
		bool finished = mSimulatorP->finished();
		result = Json(finished).dump();
	} else {
		// Invalid arguments - send error but don't return it (it's already sent)
		sendBytes(INVALID_ARGUMENTS);
		// Return empty string - error already sent via sendBytes
		return "";
	}

	return result;
}

std::string AdaptInterface::cmdGetState(const std::vector<std::string>& args) {
	std::string result = "";

	if (args.empty()) {
		dart::sim::TeamState state = mSimulatorP->getState();
		Json jsonState = convertTeamStateToJson(state);
		result = jsonState.dump();
	} else {
		// Invalid arguments - send error but don't return it (it's already sent)
		sendBytes(INVALID_ARGUMENTS);
		// Return empty string - error already sent via sendBytes
		return "";
	}

	return result;
}

std::string AdaptInterface::cmdReadForwardThreatSensor(const std::vector<std::string>& args) {
	std::string result = "";

	if (args.size() == 1) {
		unsigned cell = stoul(args[0]);
		std::vector<bool> threats = mSimulatorP->readForwardThreatSensor(cell);
		Json jsonThreats = Json(threats);
		result = jsonThreats.dump();
	} else {
		sendBytes(INVALID_ARGUMENTS);
	}

	return result;
}


std::string AdaptInterface::cmdReadForwardTargetSensor(const std::vector<std::string>& args) {
	std::string result = "";

	if (args.size() == 1) {
		unsigned cell = stoul(args[0]);
		std::vector<bool> threats = mSimulatorP->readForwardTargetSensor(cell);
		Json jsonThreats = Json(threats);
		result = jsonThreats.dump();
	} else {
		sendBytes(INVALID_ARGUMENTS);
	}

	return result;
}

std::string AdaptInterface::cmdReadForwardTargetSensorForObservations(const std::vector<std::string>& args) {
	std::string result = "";

	if (args.size() == 2) {
		unsigned cell = stoul(args[0]);
		unsigned observationCount = stoul(args[1]);
		std::vector<std::vector<bool>> targets = mSimulatorP->readForwardTargetSensor(cell, observationCount);
		Json jsonThreats = Json(targets);
		result = jsonThreats.dump();
	} else {
		sendBytes(INVALID_ARGUMENTS);
	}

	return result;
}

std::string AdaptInterface::cmdReadForwardThreatSensorForObservations(const std::vector<std::string>& args) {
	std::string result = "";

	if (args.size() == 2) {
		unsigned cell = stoul(args[0]);
		unsigned observationCount = stoul(args[1]);
		std::vector<std::vector<bool>> threats = mSimulatorP->readForwardTargetSensor(cell, observationCount);
		Json jsonThreats = Json(threats);
		result = jsonThreats.dump();
	} else {
		sendBytes(INVALID_ARGUMENTS);
	}

	return result;
}

std::string AdaptInterface::cmdStep(const std::vector<std::string>& args) {
	std::string result = "";

	std::set<std::string> tacticSet;
	unsigned index = 0;
	std::string decisionTimeMsecStr = args[args.size() - 1];
	double decisionTimeMsec = atof(decisionTimeMsecStr.c_str());

	while (index < args.size() - 1) {
		std::string tactic = args[index];
		if ( tactic[0] == '"' ) {
		    tactic.erase( 0, 1 ); // erase the first character
		    tactic.erase( tactic.size() - 1 ); // erase the last character
		}
#if DEBUG_ADAPT_INTERFACE
		std::cout << "tactic = " << tactic << std::endl;
#endif
		tacticSet.insert(tactic);

		++index;
	}

	bool stepResult = mSimulatorP->step(tacticSet, decisionTimeMsec);
	result = Json(stepResult).dump();

	return result;
}

std::string AdaptInterface::cmdGetResults(const std::vector<std::string>& args) {
	std::string result = "";

	if (args.empty()) {
		dart::sim::SimulationResults simResults = mSimulatorP->getResults();
		Json jsonSimResults = convertSimulationResultsToJson(simResults);
		result = jsonSimResults.dump();
	} else {
		sendBytes(INVALID_ARGUMENTS);
	}

	return result;
}

std::string AdaptInterface::cmdGetScreenOutput(const std::vector<std::string>& args) {
	std::string result = "";

	if (args.empty()) {
		std::string output = mSimulatorP->getScreenOutput();
		Json jsonOutput = Json(output);
		result = jsonOutput.dump();
	} else {
		sendBytes(INVALID_ARGUMENTS);
	}

	return result;
}

std::string AdaptInterface::cmdGetParameters(const std::vector<std::string>& args) {
	std::string result = "";

	if (args.empty()) {
		dart::sim::SimulationParams simParams = mSimulatorP->getParameters();
		Json jsonSimParams = convertSimulationParamsToJson(simParams);
		result = jsonSimParams.dump();
	} else {
		sendBytes(INVALID_ARGUMENTS);
	}

	return result;
}

Json AdaptInterface::convertSimulationResultsToJson(const dart::sim::SimulationResults& simResults) const {
	double decisionTimeAvg = -1;
	double decisionTimeVar = -1;

	if (!std::isnan(simResults.decisionTimeAvg)) {
		decisionTimeAvg = simResults.decisionTimeAvg;
	}

	if (!std::isnan(simResults.decisionTimeVar)) {
		decisionTimeVar = simResults.decisionTimeVar;
	}

	Json jsonState = Json::object {
		{"destroyed", simResults.destroyed},
		{"destruction positionX", simResults.whereDestroyed.x},
		{"destruction positionY", simResults.whereDestroyed.y},
		{"targetsDetected", int(simResults.targetsDetected)},
		{"missionSuccess", simResults.missionSuccess},
		{"decisionTimeAvg", decisionTimeAvg},
		{"decisionTimeVar", decisionTimeVar}
	};

	return jsonState;
}

Json AdaptInterface::convertTeamStateToJson(const dart::sim::TeamState& state) const {
	Json jsonState = Json::object {
		{"positionX", state.position.x},
		{"positionY", state.position.y},
		{"directionX", state.directionX},
		{"directionY", state.directionY},
		{"altitudeLevel", int(state.config.altitudeLevel)},
		{"formation", state.config.formation},
		{"ecm", state.config.ecm},
		{"ttcIncAlt", int(state.config.ttcIncAlt)},
		{"ttcDecAlt", int(state.config.ttcDecAlt)},
		{"ttcIncAlt2", int(state.config.ttcIncAlt2)},
		{"ttcDecAlt2", int(state.config.ttcDecAlt2)}
	};

	return jsonState;
}

Json AdaptInterface::convertSimulationParamsToJson(const dart::sim::SimulationParams& simParams) const {
	Json jsonState = Json::object {
			{"mapSize", int(simParams.mapSize)},
			{"squareMap", bool(simParams.squareMap)},
			{"altitudeLevels", int(simParams.altitudeLevels)},
			{"changeAltitudeLatencyPeriods", int(simParams.changeAltitudeLatencyPeriods)},
			{"optimalityTest", bool(simParams.optimalityTest)},
			{"threatSensorFPR", double(simParams.longRangeSensor.threatSensorFPR)},
			{"threatSensorFNR", double(simParams.longRangeSensor.threatSensorFNR)},
			{"targetSensorFPR", double(simParams.longRangeSensor.targetSensorFPR)},
			{"targetSensorFNR", double(simParams.longRangeSensor.targetSensorFNR)},
			{"targetDetectionFormationFactor", double(simParams.downwardLookingSensor.targetDetectionFormationFactor)},
			{"targetSensorRange", int(simParams.downwardLookingSensor.targetSensorRange)},
			{"destructionFormationFactor", double(simParams.threat.destructionFormationFactor)},
			{"threatRange", int(simParams.threat.threatRange)},
		};

	return jsonState;
}
}
}
