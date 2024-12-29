var serverURL = 'http://127.0.0.1:5000';//'http://100.65.118.41:5000';

function checkIfConnected() {
    fetch(`${serverURL}/ping`)
    .then(response => response.json())
    .then(data => {
        console.log(data);
    })
    .catch(error => {
        console.error('Error fetching tile data:', error);
    });
}
// setInterval(checkIfConnected, 5000);

async function getRoadsData() {
    return fetch(`${serverURL}/getRoads`)
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
    return fetch(`${serverURL}/getTile/${x}/${y}`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error fetching tile data:', error);
    });
}

async function setRoad(x, y, player) {
    return fetch(`${serverURL}/setRoad/${x}/${y}/${player}`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error setting road:', error);
    });
}

async function setSettlement(x, y, player, level) {
    return fetch(`${serverURL}/setSettlement/${x}/${y}/${player}/${level}`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error setting settlement:', error);
    });
}

async function getSettlementData() {
    return fetch(`${serverURL}/getSettlements`)
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
    return fetch(`${serverURL}/getPossibleRoads/${p}`)
    .then(response => response.json())
    .then(data => {
        console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error getting road availability:', error);
    });
}

async function getSettlementAvailability(p) {
    return fetch(`${serverURL}/getPossibleSettlements/${p}`)
    .then(response => response.json())
    .then(data => {
        console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error getting settlement availability:', error);
    });
}

async function getMaterials(p) {
    return fetch(`${serverURL}/getMaterials/${p}`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error getting materials:', error);
    });
}

async function endTurn() {
    return fetch(`${serverURL}/endTurn`)
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
    return fetch(`${serverURL}/playUntilPlayer/${p}`)
    .then(response => response.json())
    .then(data => {
        // console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error playing untill:', error);
    });
}