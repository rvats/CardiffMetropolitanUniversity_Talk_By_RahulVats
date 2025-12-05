# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:00:31 2017

@author: hkarimi
"""

class PriorityOrderRouter:
    def __init__(self, priorityDict):
        self.data = {} # data[key] = (priority, price, size)
        self.updatePriorities(priorityDict)
        
    def updateMarketData(self, exchange, price, size):
        self.data[exchange] = [self.data[exchange][0], price, size]
        
    def routeOrder(self, totalOrderSize):
        # Return dict of exchange + size ranked first by lowest price, then break ties by lowest priority
        self.order = {}
        self.currentOrderSize = totalOrderSize
        prices = self.orderedPriceList()
        for price in prices:
            self.makeOrder(price)
        return self.order
        
    def orderedPriceList(self):
        # return an ordered price list
        prices = [val[1] for key,val in self.data.items() if val[1] is not None]
        prices = list(set(prices)) # remove duplicates
        prices.sort()
        return prices
        
    def makeOrder(self, price):
        # pick up all exchanges at a given price
        exchanges = [exchange for exchange, _ in self.data.items() if self.data[exchange][1] == price]
            # sort according to lowest priority
        exchanges.sort(key=lambda x: self.data[x][1])
        for exchange in exchanges:
            self.order[exchange] = min(self.data[exchange][2], self.currentOrderSize)
            self.currentOrderSize = self.currentOrderSize - self.order[exchange]
            
    def updatePriorities(self, priorityDict):
        for exchange, priority in priorityDict.items():
            if exchange in self.data:
                self.data[exchange][0] = priority
            else:
                self.data[exchange] = (priorityDict[exchange], None, None)
                    
def main():
    priorityDict = { 'NYSE': 0, 'NASDAQ': 1, 'BATS': 3, 'ARCA': 4 }
    router = PriorityOrderRouter(priorityDict)

    # Other thread calls updateMarketData as exchange market data changes
    # Let's say final state looks like: NYSE: 300 @ 10.01, NASDAQ: 200 @ 10.00, BATS: 200 @ 10.00, ARCA: no callbacks so far
    market_data = {'NYSE': (10.01, 300), 'NASDAQ': (10.00, 200), 'BATS': (10.00, 200)}
    for key, val in market_data.items():
        router.updateMarketData(key, *val)
    result = router.routeOrder(500)
    print(result)
    # result = { 'NYSE': 100, 'NASDAQ': 200, 'BATS': 200 }
    
if __name__ == '__main__':
    main()