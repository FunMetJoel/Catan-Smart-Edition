var serverURL = "";
var httpType = "http";
var gameStartTimestamp = 0;
var lastRoll = 0;

function connectToServer(serverurl) {
    
    console.log("Connecting to server: " + serverurl + " with " + httpType);
    fetch(`${httpType}://${serverurl}/ping`)
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
        if (httpType == "https") {
            httpType = "http";
        } else {
            httpType = "https";
            connectToServer(serverurl);
        }
    });

}

async function getGameStartTimestamp() {
    return fetch(`${httpType}://${serverURL}/getGameStartTimestamp`)
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
    return fetch(`${httpType}://${serverURL}/getLastRoll`)
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
    return fetch(`${httpType}://${serverURL}/getRoads`)
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
    return fetch(`${httpType}://${serverURL}/getTile/${x}/${y}`)
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
    return fetch(`${httpType}://${serverURL}/setRoad/${x}/${y}`)
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
    return fetch(`${httpType}://${serverURL}/setSettlement/${x}/${y}`)
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
    return fetch(`${httpType}://${serverURL}/setCity/${x}/${y}`)
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
    return fetch(`${httpType}://${serverURL}/getSettlements`)
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
    return fetch(`${httpType}://${serverURL}/getPossibleRoads/${p}`)
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
    return fetch(`${httpType}://${serverURL}/getPossibleSettlements/${p}`)
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
    return fetch(`${httpType}://${serverURL}/getMaterials/${p}`)
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
    return fetch(`${httpType}://${serverURL}/getPoints/${p}`)
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
    return fetch(`${httpType}://${serverURL}/endTurn`)
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
    return fetch(`${httpType}://${serverURL}/playUntilPlayer/${p}`)
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
    return fetch(`${httpType}://${serverURL}/makeAIMove`)
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
    return fetch(`${httpType}://${serverURL}/getCurrentPlayer`)
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
    return fetch(`${httpType}://${serverURL}/resetGameIfOver`)
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
    return fetch(`${httpType}://${serverURL}/getRobberData`)
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
    return fetch(`${httpType}://${serverURL}/trade/${give}/${get}`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error trading:', error);
    });
}