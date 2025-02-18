var serverURL = "";
var gameStartTimestamp = 0;
var lastRoll = 0;

function connectToServer(serverurl) {
    
    console.log("Connecting to server: " + serverurl);

    fetch(`http://${serverurl}/ping`)
    .then(response => response.json())
    .then(
        response => {
            if (response == "pong") {
                console.log("Connected to server: " + serverurl);

                document.getElementById("connectDiv").style.display = "none";
                serverURL = serverurl;

            } else {
                console.error("Failed to connect to server: " + serverurl);
            }
        }
    )
    .catch(error => {
        console.error('Error fetching tile data:', error);
    });

}

async function getGameStartTimestamp() {
    return fetch(`http://${serverURL}/getGameStartTimestamp`)
    .then(response => response.json())
    .then(data => {
        // console.log(Date.now(), data);
        gameStartTimestamp = data;
        return data;
    })
    .catch(error => {
        console.error('Error fetching tile data:', error);
    });
}

async function getLastRoll() {
    return fetch(`http://${serverURL}/getLastRoll`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        lastRoll = data;
        return data;
    })
    .catch(error => {
        console.error('Error fetching tile data:', error);
    });
}

async function getRoadsData() {
    return fetch(`http://${serverURL}/getRoads`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error fetching tile data:', error);
    });
}

async function getTileData(x, y) {
    return fetch(`http://${serverURL}/getTile/${x}/${y}`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error fetching tile data:', error);
    });
}

async function setRoad(x, y) {
    return fetch(`http://${serverURL}/setRoad/${x}/${y}`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error setting road:', error);
    });
}

async function setSettlement(x, y) {
    return fetch(`http://${serverURL}/setSettlement/${x}/${y}`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error setting settlement:', error);
    });
}

async function setCity(x, y) {
    return fetch(`http://${serverURL}/setCity/${x}/${y}`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error setting city:', error);
    });
}

async function getSettlementData() {
    return fetch(`http://${serverURL}/getSettlements`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error fetching settlement data:', error);
    });
}

async function getRoadAvailability(p) {
    return fetch(`http://${serverURL}/getPossibleRoads/${p}`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error getting road availability:', error);
    });
}

async function getSettlementAvailability(p) {
    return fetch(`http://${serverURL}/getPossibleSettlements/${p}`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error getting settlement availability:', error);
    });
}

async function getMaterials(p) {
    return fetch(`http://${serverURL}/getMaterials/${p}`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error getting materials:', error);
    });
}

async function getPoints(p) {
    return fetch(`http://${serverURL}/getPoints/${p}`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error getting points:', error);
    });
}

async function endTurn() {
    return fetch(`http://${serverURL}/endTurn`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error ending turn:', error);
    });
}

async function playUntill(p) {
    return fetch(`http://${serverURL}/playUntilPlayer/${p}`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error playing untill:', error);
    });
}

async function aiPlay() {
    return fetch(`http://${serverURL}/makeAIMove`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error AI playing:', error);
    });
}

async function getCurrentPlayer() {
    return fetch(`http://${serverURL}/getCurrentPlayer`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error getting current player:', error);
    });
}

async function resetGameIfOver() {
    return fetch(`http://${serverURL}/resetGameIfOver`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error resetting game:', error);
    });
}

async function getRobberData() {
    return fetch(`http://${serverURL}/getRobberData`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error getting robber data:', error);
    });
}

async function trade(give, get) {
    return fetch(`http://${serverURL}/trade/${give}/${get}`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error trading:', error);
    });
}