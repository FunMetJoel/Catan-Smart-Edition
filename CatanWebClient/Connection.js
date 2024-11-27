var serverURL = 'http://100.65.118.41:5000';

function checkIfConnected() {
    fetch(`${serverURL}/ping`)
    .then(response => response.json())
    .then(data => {
        console.log(data);
    })
    .catch(error => {
        console.error('Error fetching tile data:', error);
    });

    fetch(`${serverURL}/getRoads`)
    .then(response => response.json())
    .then(data => {
        console.log(data);
    })
    .catch(error => {
        console.error('Error fetching tile data:', error);
    });
}
setInterval(checkIfConnected, 5000);

async function getTileData(x, y) {
    return fetch(`${serverURL}/getTile/${x}/${y}`)
    .then(response => response.json())
    .then(data => {
        console.log(data);
        return data;
    })
    .catch(error => {
        console.error('Error fetching tile data:', error);
    });
}