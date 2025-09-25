"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Send, Sparkles, User, Bot, TrendingUp, Heart, ChevronDown, MessageSquare, Plus, MoreHorizontal, Edit2, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { CopilotChat } from "@copilotkit/react-ui"
import { useCopilotChat } from "@copilotkit/react-core"
import { ToolLog } from "./tool-logs"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuSeparator } from "@/components/ui/dropdown-menu"
interface Message {
  id: string
  type: "user" | "assistant"
  content: string
  timestamp: Date
}

interface Product {
  id: string
  name: string
  price: string
  image: string
  pros: string[]
  cons: string[]
  source: string
  rating: number
  reviews: number
  specs: {
    processor: string
    ram: string
    storage: string
    display: string
    battery: string
    weight: string
    ports: string
    os: string
  }
}

interface ChatSession {
  id: number
  name: string
  timestamp: Date
  messageCount: number
  lastActivity: Date
}

interface SidebarProps {
  setQuery: (query: string) => void
  clearState: (state: any) => void
  onSearch: (query: string) => void
  suggestions: string[]
  currentQuery: string
  isSearching: boolean
  currentView: "products" | "wishlist" | "report"
  wishlistCount: number
  goToProducts: () => void
  currentChatId: number
  isLoading: boolean
  chatSessions: any[]
  onSwitchChat: (chatId: number) => void
  onCreateNewChat: () => void
  onRenameChat: (chatId: number, newName: string) => void
  onDeleteChat: (chatId: number) => void
}

const initialMessages: Message[] = [
  {
    id: "1",
    type: "assistant",
    content:
      "Hi! I'm your AI shopping assistant. I can help you find and compare products across multiple websites. What are you looking for today?",
    timestamp: new Date(),
  },
]

const quickSuggestions = ["Compare these laptops", "Which has better battery life?", "Show me cheaper alternatives"]

export function Sidebar({
  setQuery,
  isLoading,
  clearState,
  onSearch,
  suggestions,
  currentQuery,
  isSearching,
  currentView,
  wishlistCount,
  goToProducts,
  currentChatId,
  chatSessions,
  onSwitchChat,
  onCreateNewChat,
  onRenameChat,
  onDeleteChat,
}: SidebarProps) {
  const [editingChatId, setEditingChatId] = useState<number | null>(null)
  const [editingName, setEditingName] = useState("")
  useEffect(() => {
    debugger
    console.log(chatSessions, "chatSessions");
    
    // setCurrentChatId(chatSessions[0]?.conversationId)
  }, [])
  // let currentChat = null;
  // if(chatSessions && Array.isArray(chatSessions)){
  
  const currentChat = chatSessions?.find(chat => chat.conversationId === currentChatId)
  // }
  
  const formatTimeAgo = (date: Date) => {
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)
    
    if (diffMins < 1) return "Just now"
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`
    return date.toLocaleDateString()
  }

  const handleRenameSubmit = (chatId: number) => {
    if (editingName.trim()) {
      onRenameChat(chatId, editingName.trim())
    }
    setEditingChatId(null)
    setEditingName("")
  }

  const startEditing = (chat:any) => {
    setEditingChatId(chat.conversationId)
    setEditingName(chat.chatName)
  }




  const generateComparisonAnalysis = (products: Product[]) => {
    const productNames = products.map((p) => p.name).join(", ")

    return `## Comparison Analysis: ${productNames}

**💰 Price Comparison:**
${products.map((p) => `• ${p.name}: ${p.price}`).join("\n")}

**⭐ Rating Analysis:**
${products.map((p) => `• ${p.name}: ${p.rating}/5 (${p.reviews} reviews)`).join("\n")}

**🔋 Battery Life:**
${products.map((p) => `• ${p.name}: ${p.specs.battery}`).join("\n")}

**💻 Performance:**
${products.map((p) => `• ${p.name}: ${p.specs.processor}`).join("\n")}

**📊 My Recommendation:**
${products.length === 2
        ? `Based on the comparison, the ${products[0].rating > products[1].rating ? products[0].name : products[1].name} offers better overall value with its ${products[0].rating > products[1].rating ? "higher rating and" : ""} strong performance. However, consider your budget and specific needs.`
        : `Among these ${products.length} options, I'd recommend focusing on your primary use case - whether that's performance, battery life, or value for money.`
      }

Would you like me to dive deeper into any specific aspect of this comparison?`
  }


  const getViewTitle = () => {
    switch (currentView) {
      case "report":
        return "Analysis Report"
      case "wishlist":
        return "Wishlist Mode"
      default:
        return "AI Shopping Assistant"
    }
  }



  return (
    <div className="flex flex-col min-h-screen w-80 bg-[#FAFCFA] border-r border-[#D8D8E5]">
      {/* Header */}
      <div className="p-4 border-b border-[#D8D8E5] bg-white">
        <div className="flex items-center gap-3 mb-3">
          <div className="text-2xl">🪁</div>
          <div className="flex-1">
            <h1 className="text-lg font-semibold text-[#030507] font-['Roobert']">Shopping Assistant</h1>
            <Badge variant="secondary" className="text-xs bg-[#BEC9FF] text-[#030507] font-semibold">
              PRO
            </Badge>
          </div>
        </div>
        
        {/* Chat Dropdown */}
        <div className="space-y-2">
          <DropdownMenu >
            <DropdownMenuTrigger onBlur={() => {setEditingChatId(null), setEditingName("")}} disabled={isLoading} asChild>
              <Button 
                variant="outline" 
                className="w-full justify-between bg-[#F8F9FD] border-[#E8E8EF] hover:bg-[#F0F1F7] text-left h-auto p-3"
              >
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  <MessageSquare className="w-4 h-4 text-[#6B7280] flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-[#030507] truncate">
                      {currentChat?.chatName || "New Chat"}
                    </div>
                    <div className="text-xs text-[#6B7280]">
                      {currentChat ? `${currentChat.messages.length} messages` : "Start a conversation"}
                    </div>
                  </div>
                </div>
                <ChevronDown className="w-4 h-4 text-[#6B7280] flex-shrink-0" />
              </Button>
            </DropdownMenuTrigger>
            
            <DropdownMenuContent align="start" className="w-72 bg-white border-[#D8D8E5] shadow-lg">
              {/* New Chat Option */}
              <DropdownMenuItem 
                onClick={onCreateNewChat}
                className="flex items-center gap-3 p-3 hover:bg-[#F8F9FD] cursor-pointer"
              >
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#86ECE4] to-[#BEC9FF] flex items-center justify-center flex-shrink-0">
                  <Plus className="w-4 h-4 text-[#030507]" />
                </div>
                <div className="flex-1">
                  <div className="font-medium text-[#030507]">New Chat</div>
                  <div className="text-xs text-[#6B7280]">Start a fresh conversation</div>
                </div>
              </DropdownMenuItem>
              
              <DropdownMenuSeparator className="bg-[#E8E8EF]" />
              
              {/* Chat History */}
              <div className="max-h-64 overflow-y-auto">
                {chatSessions && Array.isArray(chatSessions) ? chatSessions.map((chat) => (
                  <div key={chat.conversationId} className="group">
                    {editingChatId === chat.conversationId ? (
                      <div className="p-3 flex items-center gap-2">
                        <MessageSquare className="w-4 h-4 text-[#6B7280] flex-shrink-0" />
                        <Input
                          value={editingName}
                          onChange={(e) => setEditingName(e.target.value)}
                          onKeyDown={(e) => {
                            debugger
                            if (e.key === 'Enter') handleRenameSubmit(chat.conversationId)
                            if (e.key === 'Escape') {
                              setEditingChatId(null)
                              setEditingName("")
                            }
                          }}

                          className="flex-1 h-6 text-sm"
                          autoFocus
                        />
                      </div>
                    ) : (
                      <DropdownMenuItem 
                        onClick={() => onSwitchChat(chat.conversationId)}
                        className="flex items-center gap-3 p-3 hover:bg-[#F8F9FD] cursor-pointer relative group"
                      >
                        <div className={`w-2 h-2 rounded-full flex-shrink-0 ${
                          chat.conversationId === currentChatId ? 'bg-[#86ECE4]' : 'bg-[#E8E8EF]'
                        }`} />
                        <div className="flex-1 min-w-0">
                          <div className={`font-medium truncate ${
                            chat.conversationId === currentChatId ? 'text-[#030507]' : 'text-[#6B7280]'
                          }`}>
                            {chat.chatName}
                          </div>
                          <div className="text-xs text-[#9CA3AF]">
                            {chat.messages.length} messages 
                          </div>
                        </div>
                        
                        {/* Action Menu */}
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="opacity-0 group-hover:opacity-100 h-6 w-6 p-0 hover:bg-[#E8E8EF]"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <MoreHorizontal className="w-3 h-3" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end" className="w-40">
                            <DropdownMenuItem 
                              onClick={(e) => {
                                debugger
                                e.stopPropagation()
                                startEditing(chat)
                              }}
                              className="flex items-center gap-2"
                            >
                              <Edit2 className="w-3 h-3" />
                              Rename
                            </DropdownMenuItem>
                            <DropdownMenuItem 
                              onClick={(e) => {
                                e.stopPropagation()
                                onDeleteChat(chat.conversationId)
                              }}
                              className="flex items-center gap-2 text-red-600 hover:text-red-700 hover:bg-red-50"
                            >
                              <Trash2 className="w-3 h-3" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </DropdownMenuItem>
                    )}
                  </div>
                )) : null}
              </div>
              
              {(!chatSessions || !Array.isArray(chatSessions) || chatSessions.length === 0) && (
                <div className="p-4 text-center text-[#6B7280] text-sm">
                  No chat history yet
                </div>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
          
          {/* View Status */}
          <div className="text-xs text-[#6B7280] px-1">
            {getViewTitle()}
          </div>
        </div>
      </div>
      <div className="flex-1 overflow-auto">
        <CopilotChat className="h-full" labels={{
          initial: "Hi! I'm your AI shopping assistant. I can help you find and compare products across multiple websites. What are you looking for today? Laptops, Phones, Headphones, etc.",
        }} />
      </div>
    </div>
  )
}
